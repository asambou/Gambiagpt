import streamlit as st
import requests
import feedparser
import ipaddress
import folium
from streamlit_folium import st_folium
from supabase import create_client
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from tavily import TavilyClient
import bcrypt

VECTOR_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are GambiaGPT, the most knowledgeable AI assistant about The Gambia AND a world-class cybersecurity and networking tutor.

GAMBIA KNOWLEDGE:
- Gambian history, politics, government, culture, traditions
- Geography, economy, tourism, health, education
- Current events and news

CYBERSECURITY & NETWORKING EXPERTISE:
- Routing and switching (CCNA level and beyond)
- OSI model, TCP/IP, subnetting, VLANs, STP, OSPF, BGP, EIGRP
- Cisco IOS commands and configuration
- Cybersecurity fundamentals, CEH, CompTIA Security+
- Ethical hacking, penetration testing, Kali Linux tools
- Firewalls, VPNs, IDS/IPS, network security
- Python for networking and security automation
- CTF challenges and career guidance in Africa

LANGUAGE RULES:
- Detect the language the user writes in automatically.
- Reply in Mandinka, Wolof, Fula, or Jola if they write in those languages.
- Reply in English for all other languages.

ANSWER STYLE:
- Be confident, warm, and detailed like a senior network engineer and Gambian scholar.
- For technical questions give real commands, configs, and examples.
- Never say the context does not mention — just answer naturally.
- Always encourage Gambian youth in tech careers.
"""

# ── HELPERS ──
@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})
@st.cache_resource
def load_legal_retriever():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.load_local(
            "vectorstore_legal",
            embeddings,
            allow_dangerous_deserialization=True
        )
        return db.as_retriever(search_kwargs={"k": 4})
    except:
        return load_retriever()


def web_search(query):
    try:
        tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        results = tavily.search(query=query, max_results=5)
        texts = [r["content"] for r in results.get("results", [])]
        return "\n\n".join(texts)[:2000]
    except:
        return ""

def get_answer(query):
    try:
        retriever = load_retriever()
        docs = retriever.invoke(query)
        doc_context = "\n\n".join(doc.page_content for doc in docs)[:600]
        web_context = web_search(query)
        combined_context = f"WEB:\n{web_context}\n\nDOCS:\n{doc_context}"
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"], temperature=0.3)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": combined_context, "question": query})
    except:
        return "Sorry, something went wrong. Please try again."
    
def get_legal_answer(query):
    try:
        retriever = load_legal_retriever()
        docs = retriever.invoke(query)

        context_with_sources = ""
        sources = []
        for doc in docs:
            context_with_sources += doc.page_content + "\n\n"
            source = doc.metadata.get("source", "Gambian Law")
            if source not in sources:
                sources.append(source)

        context_with_sources = context_with_sources[:2000]

        web_context = web_search(f"{query} Gambia law legislation")

        combined = f"LEGAL DOCUMENTS:\n{context_with_sources}\n\nWEB RESEARCH:\n{web_context}"

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.1
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Gambian legal expert and constitutional scholar.
Answer legal questions based on Gambian law.
Always cite the specific law, act, or constitutional article you are referencing.
Be precise, clear, and use plain language so ordinary citizens understand.
Format your answer with:
1. Direct answer
2. Legal basis (which law or article)
3. Practical implications
If you are not certain, say so clearly."""),
            ("human", "Legal question: {question}\n\nContext from Gambian laws:\n{context}")
        ])

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": query, "context": combined})

        if sources:
            clean_sources = [s.split("/")[-1] for s in sources]
            answer += f"\n\n📚 **Sources:** {', '.join(clean_sources)}"

        return answer
    except Exception as e:
        return get_answer(query)

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(email, password):
    try:
        supabase = get_supabase()
        existing = supabase.table("users").select("id").eq("email", email).execute()
        if existing.data:
            return False, "Email already registered."
        hashed = hash_password(password)
        supabase.table("users").insert({"email": email, "password_hash": hashed}).execute()
        return True, "Account created!"
    except Exception as e:
        return False, str(e)

def login_user(email, password):
    try:
        supabase = get_supabase()
        result = supabase.table("users").select("*").eq("email", email).execute()
        if not result.data:
            return False, None, "Email not found."
        user = result.data[0]
        if check_password(password, user["password_hash"]):
            return True, user, "Login successful!"
        return False, None, "Wrong password."
    except Exception as e:
        return False, None, str(e)

def save_message(user_id, role, content):
    try:
        get_supabase().table("chat_history").insert({"user_id": user_id, "role": role, "content": content}).execute()
    except:
        pass

def load_history(user_id):
    try:
        result = get_supabase().table("chat_history").select("*").eq("user_id", user_id).order("created_at").limit(50).execute()
        return result.data or []
    except:
        return []

def clear_history(user_id):
    try:
        get_supabase().table("chat_history").delete().eq("user_id", user_id).execute()
    except:
        pass

def subnet_calculator(ip, prefix):
    try:
        network = ipaddress.IPv4Network(f"{ip}/{prefix}", strict=False)
        hosts = list(network.hosts())
        return {
            "network": str(network.network_address),
            "broadcast": str(network.broadcast_address),
            "netmask": str(network.netmask),
            "wildcard": str(network.hostmask),
            "hosts": network.num_addresses - 2,
            "first_host": str(hosts[0]) if hosts else "N/A",
            "last_host": str(hosts[-1]) if hosts else "N/A",
            "ip_class": "A" if int(ip.split(".")[0]) < 128 else "B" if int(ip.split(".")[0]) < 192 else "C"
        }
    except:
        return None

# ── PAGE CONFIG ──
st.set_page_config(page_title="GambiaGPT", page_icon="🇬🇲", layout="wide")

# ── SIDEBAR ──
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/77/Flag_of_The_Gambia.svg", width=80)
    st.title("🇬🇲 GambiaGPT")
    st.caption("Your AI guide to The Gambia")
    st.divider()

    page = st.radio("Navigate:", [
        "💬 Chat",
        "📰 News",
        "🗺️ Map",
        "🎓 Education",
        "🔐 Cybersecurity",
        "🌐 Networking",
        "📞 Emergency",
        "⚖️ Legal & Law",
    ])

    st.divider()
    st.header("👤 Account")

    if "user" not in st.session_state:
        st.session_state.user = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.user:
        st.success(f"✅ {st.session_state.user['email']}")
        if st.button("Load history"):
            history = load_history(st.session_state.user["id"])
            st.session_state.messages = [{"role": h["role"], "content": h["content"]} for h in history]
            st.rerun()
        if st.button("Clear history"):
            clear_history(st.session_state.user["id"])
            st.session_state.messages = []
            st.rerun()
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.messages = []
            st.rerun()
    else:
        auth_tab = st.radio("", ["Login", "Register"], horizontal=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if auth_tab == "Login":
            if st.button("Login", type="primary"):
                if email and password:
                    success, user, msg = login_user(email, password)
                    if success:
                        st.session_state.user = user
                        history = load_history(user["id"])
                        st.session_state.messages = [{"role": h["role"], "content": h["content"]} for h in history]
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        else:
            if st.button("Create account", type="primary"):
                if email and password:
                    if len(password) < 6:
                        st.error("Password must be 6+ characters.")
                    else:
                        success, msg = register_user(email, password)
                        st.success(msg) if success else st.error(msg)

    st.divider()
    st.caption("Built for Gambia 🇬🇲 | Free to use")

# ════════════════════════════════════════
# ── PAGE: CHAT ──
# ════════════════════════════════════════
if page == "💬 Chat":
    st.title("💬 Ask GambiaGPT")
    st.info("Ask in English, Mandinka, Wolof, Jola or Fula — powered by live web search.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask anything about Gambia, cybersecurity, or networking..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(query)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if st.session_state.user:
            save_message(st.session_state.user["id"], "user", query)
            save_message(st.session_state.user["id"], "assistant", answer)

# ════════════════════════════════════════
# ── PAGE: NEWS ──
# ════════════════════════════════════════
elif page == "📰 News":
    st.title("📰 Gambia News Digest")
    news_sources = {
        "The Point": "https://thepoint.gm",
        "Foroyaa": "https://foroyaa.net",
        "Gainako": "https://gainako.com",
        "The Standard": "https://thestandard.gm",
        "WhatsOn Gambia": "https://whatson-gambia.com",
        "Kerr Fatou": "https://www.kerrfatou.com",
        "Fatu Network": "https://fatunetwork.net",
        "SMBC News": "https://smbcnewsgambia.com",
    }

    col1, col2 = st.columns(2)
    with col1:
        selected_source = st.selectbox("Source:", list(news_sources.keys()))
        category = st.selectbox("Category:", ["Latest news", "Politics", "Business", "Health", "Education", "Sports", "Technology"])
    with col2:
        num_stories = st.slider("Stories:", 3, 10, 5)
        briefing_topic = st.text_input("AI briefing topic:", placeholder="e.g. Gambia economy")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Fetch news", type="primary"):
            with st.spinner("Fetching..."):
                try:
                    tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
                    results = tavily.search(query=f"Gambia {category} {news_sources[selected_source]}", max_results=num_stories)
                    for i, article in enumerate(results.get("results", []), 1):
                        with st.expander(f"📰 {i}. {article.get('title', 'No title')}"):
                            st.write(article.get("content", "")[:400] + "...")
                            if article.get("url"):
                                st.markdown(f"[Read more]({article['url']})")
                except Exception as e:
                    st.error(str(e))
    with col_b:
        if st.button("Generate AI briefing", type="primary"):
            if briefing_topic:
                with st.spinner("Writing briefing..."):
                    answer = get_answer(f"Write a professional news briefing about: {briefing_topic} in Gambia. Include latest developments, key facts, and analysis.")
                    st.markdown(answer)

    st.divider()
    st.subheader("⚡ Quick topics")
    quick_topics = ["Gambia economy", "Adama Barrow", "Gambia health", "Gambia education", "Gambia football", "Gambia tourism"]
    cols = st.columns(3)
    for i, topic in enumerate(quick_topics):
        with cols[i % 3]:
            if st.button(f"📰 {topic}", key=f"news_{i}"):
                with st.spinner("Loading..."):
                    st.markdown(get_answer(f"Latest news about: {topic}"))

    st.divider()
    st.subheader("📡 Gambian Media")
    st.markdown("""
| Media | Website | Facebook |
|-------|---------|----------|
| The Point | [thepoint.gm](https://thepoint.gm) | facebook.com/thepointgambia |
| Foroyaa | [foroyaa.net](https://foroyaa.net) | facebook.com/foroyaa |
| WhatsOn Gambia | [whatson-gambia.com](https://whatson-gambia.com) | facebook.com/whatson.gambia |
| Kerr Fatou | [kerrfatou.com](https://www.kerrfatou.com) | facebook.com/kerrfatou |
| Fatu Network | [fatunetwork.net](https://fatunetwork.net) | facebook.com/fatunetwork |
| GRTS | [grts.gm](https://grts.gm) | facebook.com/grtsgambia |
    """)

# ════════════════════════════════════════
# ── PAGE: MAP ──
# ════════════════════════════════════════
elif page == "🗺️ Map":
    st.title("🗺️ Interactive Map of The Gambia")
    map_filter = st.multiselect("Show:", ["Tourist spots", "Hospitals", "Universities", "Hotels", "Markets", "Embassies"], default=["Tourist spots"])
    map_style = st.selectbox("Style:", ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"])

    m = folium.Map(location=[13.4549, -15.3100], zoom_start=8, tiles=map_style)

    all_places = {
        "Tourist spots": [
            {"name": "Kachikally Crocodile Pool", "lat": 13.4441, "lon": -16.6774, "desc": "Sacred crocodile pool in Bakau."},
            {"name": "Abuko Nature Reserve", "lat": 13.3833, "lon": -16.6500, "desc": "Gambia's first nature reserve."},
            {"name": "Wassu Stone Circles", "lat": 13.6833, "lon": -14.9167, "desc": "UNESCO World Heritage Site."},
            {"name": "Arch 22", "lat": 13.4533, "lon": -16.5756, "desc": "Iconic monument in Banjul."},
            {"name": "James Island", "lat": 13.4833, "lon": -16.5333, "desc": "UNESCO World Heritage Site."},
            {"name": "Bijilo Forest Park", "lat": 13.4167, "lon": -16.7167, "desc": "Coastal forest with monkeys."},
        ],
        "Hospitals": [
            {"name": "Royal Victoria Teaching Hospital", "lat": 13.4544, "lon": -16.5786, "desc": "Main national hospital."},
            {"name": "Serekunda General Hospital", "lat": 13.4386, "lon": -16.6775, "desc": "Largest outside Banjul."},
            {"name": "MRC Gambia", "lat": 13.4167, "lon": -16.6667, "desc": "Medical Research Council."},
        ],
        "Universities": [
            {"name": "University of The Gambia", "lat": 13.4167, "lon": -16.6500, "desc": "National university."},
            {"name": "GTTI", "lat": 13.4500, "lon": -16.6500, "desc": "Technical training institute."},
            {"name": "Gambia College", "lat": 13.3833, "lon": -16.6833, "desc": "Teacher and nursing training."},
        ],
        "Hotels": [
            {"name": "Coco Ocean Resort", "lat": 13.4167, "lon": -16.7333, "desc": "5-star luxury resort."},
            {"name": "Kairaba Beach Hotel", "lat": 13.4167, "lon": -16.7200, "desc": "Top beach hotel."},
            {"name": "Mandina Lodges", "lat": 13.5000, "lon": -15.8000, "desc": "Eco-lodge on the river."},
        ],
        "Markets": [
            {"name": "Albert Market Banjul", "lat": 13.4531, "lon": -16.5731, "desc": "Oldest famous market."},
            {"name": "Serrekunda Market", "lat": 13.4386, "lon": -16.6775, "desc": "Largest market in Gambia."},
            {"name": "Brikama Market", "lat": 13.2667, "lon": -16.6500, "desc": "Known for wood carvings."},
        ],
        "Embassies": [
            {"name": "US Embassy", "lat": 13.4533, "lon": -16.5800, "desc": "United States Embassy."},
            {"name": "UK High Commission", "lat": 13.4500, "lon": -16.5750, "desc": "British High Commission."},
        ],
    }

    colors = {"Tourist spots": "green", "Hospitals": "red", "Universities": "blue", "Hotels": "purple", "Markets": "orange", "Embassies": "darkblue"}
    icons = {"Tourist spots": "star", "Hospitals": "plus", "Universities": "graduation-cap", "Hotels": "home", "Markets": "shopping-cart", "Embassies": "flag"}

    for category in map_filter:
        if category in all_places:
            for place in all_places[category]:
                folium.Marker(
                    location=[place["lat"], place["lon"]],
                    popup=folium.Popup(f"<b>{place['name']}</b><br>{place['desc']}", max_width=250),
                    tooltip=place["name"],
                    icon=folium.Icon(color=colors[category], icon=icons[category], prefix="fa")
                ).add_to(m)

    folium.PolyLine(
        locations=[[13.4667, -16.5833], [13.5000, -15.8000], [13.5500, -14.9000], [13.5500, -14.6500]],
        color="blue", weight=3, opacity=0.6, tooltip="River Gambia"
    ).add_to(m)

    st_folium(m, width=900, height=500)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tourist spots", 6)
    col2.metric("Hospitals", 3)
    col3.metric("Universities", 3)
    col4.metric("Hotels", 3)

# ════════════════════════════════════════
# ── PAGE: EDUCATION ──
# ════════════════════════════════════════
elif page == "🎓 Education":
    st.title("🎓 University & Education Guide")

    edu_section = st.radio("Section:", ["Universities", "Admission", "Scholarships", "AI Advisor"], horizontal=True)

    if edu_section == "Universities":
        st.subheader("Universities & Colleges")
        for uni in [
            {"name": "University of The Gambia (UTG)", "type": "Public", "phone": "+220 441 2200", "fee": "GMD 15,000-45,000/yr", "web": "utm.edu.gm"},
            {"name": "Gambia College", "type": "Public", "phone": "+220 441 2345", "fee": "GMD 10,000-25,000/yr", "web": "gambiacollege.edu.gm"},
            {"name": "GTTI", "type": "Public", "phone": "+220 439 1234", "fee": "GMD 8,000-20,000/yr", "web": "gtti.edu.gm"},
            {"name": "AIUWA", "type": "Private", "phone": "+220 439 5678", "fee": "GMD 30,000-60,000/yr", "web": "aiuwa.com"},
            {"name": "MDI", "type": "Public", "phone": "+220 439 7890", "fee": "GMD 12,000-30,000/yr", "web": "mdi.edu.gm"},
        ]:
            with st.expander(f"🏛️ {uni['name']} — {uni['type']}"):
                col_a, col_b = st.columns(2)
                col_a.write(f"📞 {uni['phone']}")
                col_a.write(f"💰 {uni['fee']}")
                col_b.markdown(f"🌐 [{uni['web']}](https://{uni['web']})")

    elif edu_section == "Admission":
        st.subheader("Admission Requirements")
        uni = st.selectbox("University:", ["UTG", "Gambia College", "GTTI", "AIUWA", "MDI"])
        program = st.selectbox("Program:", ["General", "Medicine", "Engineering", "Law", "Business", "Education", "Nursing", "ICT"])
        if st.button("Get requirements", type="primary"):
            with st.spinner("Loading..."):
                st.markdown(get_answer(f"Admission requirements for {program} at {uni} Gambia. Include WASSCE grades and documents needed."))
        st.divider()
        st.markdown("""
**Documents typically needed:**
- WASSCE results
- Birth certificate
- National ID or passport
- 2 passport photos
- Recommendation letter
- Application fee receipt
        """)

    elif edu_section == "Scholarships":
        st.subheader("Scholarships for Gambian Students")
        for s in [
            {"name": "Gambia Government Scholarship", "coverage": "Full tuition + stipend", "deadline": "August", "link": "moherst.gov.gm"},
            {"name": "Commonwealth Scholarship", "coverage": "Full + flights + allowance", "deadline": "October-December", "link": "cscuk.fcdo.gov.uk"},
            {"name": "Mastercard Foundation", "coverage": "Full + accommodation + laptop", "deadline": "Varies", "link": "mastercardfdn.org"},
            {"name": "Turkiye Burslari", "coverage": "Full + flights + stipend", "deadline": "February", "link": "turkiyeburslari.gov.tr"},
            {"name": "Chinese Government (CSC)", "coverage": "Full + accommodation + stipend", "deadline": "March-April", "link": "csc.edu.cn"},
            {"name": "Islamic Development Bank", "coverage": "Tuition + living allowance", "deadline": "January", "link": "isdb.org"},
        ]:
            with st.expander(f"🏆 {s['name']}"):
                col_a, col_b = st.columns(2)
                col_a.write(f"💰 {s['coverage']}")
                col_b.write(f"📅 Deadline: {s['deadline']}")
                st.markdown(f"🌐 [Apply here](https://{s['link']})")

    elif edu_section == "AI Advisor":
        st.subheader("AI Education Advisor")
        for q in ["Best university in Gambia for medicine?", "How to apply for government scholarship?", "How to study abroad for free from Gambia?"]:
            if st.button(f"❓ {q}", key=q):
                with st.spinner("Advising..."):
                    st.markdown(get_answer(q))
        custom_q = st.text_area("Your question:")
        if st.button("Ask advisor", type="primary"):
            if custom_q:
                with st.spinner("Advising..."):
                    st.markdown(get_answer(f"As a Gambian education advisor: {custom_q}"))

# ════════════════════════════════════════
# ── PAGE: CYBERSECURITY ──
# ════════════════════════════════════════
elif page == "🔐 Cybersecurity":
    st.title("🔐 Cybersecurity Lab")
    topic = st.selectbox("Choose topic:", [
        "Pick a topic...", "CEH overview", "CompTIA Security+", "Kali Linux top 10 tools",
        "How to do a penetration test", "Types of cyberattacks", "Firewalls explained",
        "VPN explained", "Password attacks", "Social engineering and phishing",
        "Cybersecurity career in Gambia", "CTF beginner guide", "Python for security",
        "OWASP Top 10", "Nmap scanning", "Wireshark basics",
    ])
    if topic != "Pick a topic...":
        if st.button("Learn", type="primary"):
            with st.spinner("Loading..."):
                st.markdown(get_answer(f"Teach me about: {topic}. Give detailed explanation with examples and commands."))
    st.divider()
    q = st.text_input("Ask any cybersecurity question:")
    if st.button("Ask", key="cyber"):
        if q:
            with st.spinner("Thinking..."):
                st.markdown(get_answer(f"As cybersecurity expert: {q}"))
    st.divider()
    st.subheader("🎯 Roadmap")
    st.markdown("""
**Beginner:** ITF+ → A+ → Network+ → Security+

**Intermediate:** CEH → eJPT → OSCP

**Free resources:** [TryHackMe](https://tryhackme.com) | [HackTheBox](https://hackthebox.com) | [Cybrary](https://cybrary.it)
    """)

# ════════════════════════════════════════
# ── PAGE: NETWORKING ──
# ════════════════════════════════════════
elif page == "🌐 Networking":
    st.title("🌐 Networking Lab")
    net_section = st.radio("Section:", ["Study topics", "Subnet calculator", "Cisco IOS reference", "Ask engineer"], horizontal=True)

    if net_section == "Study topics":
        topic = st.selectbox("Choose topic:", [
            "Pick a topic...", "OSI model 7 layers", "TCP/IP vs OSI",
            "How routing works", "OSPF configuration", "BGP explained",
            "VLANs and trunking", "STP explained", "NAT and PAT",
            "DHCP explained", "DNS explained", "ACLs on Cisco",
            "How switches work", "IPv6 subnetting", "CCNA study guide",
        ])
        if topic != "Pick a topic...":
            if st.button("Learn", type="primary"):
                with st.spinner("Loading..."):
                    st.markdown(get_answer(f"CCNA instructor teaching: {topic}. Include Cisco commands and configs."))

    elif net_section == "Subnet calculator":
        col1, col2 = st.columns(2)
        with col1:
            ip = st.text_input("IP Address", "192.168.1.0")
        with col2:
            prefix = st.slider("Prefix", 8, 30, 24)
        if st.button("Calculate", type="primary"):
            r = subnet_calculator(ip, prefix)
            if r:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Network", r["network"])
                    st.metric("Subnet mask", r["netmask"])
                    st.metric("First host", r["first_host"])
                    st.metric("Class", r["ip_class"])
                with col_b:
                    st.metric("Broadcast", r["broadcast"])
                    st.metric("Wildcard", r["wildcard"])
                    st.metric("Last host", r["last_host"])
                    st.metric("Usable hosts", f"{r['hosts']:,}")
            else:
                st.error("Invalid IP address.")

    elif net_section == "Cisco IOS reference":
        st.markdown("""
| Task | Command |
|------|---------|
| Privileged mode | `enable` |
| Config mode | `configure terminal` |
| Set hostname | `hostname R1` |
| Show interfaces | `show ip interface brief` |
| Show routes | `show ip route` |
| Configure interface | `interface gi0/0` |
| Set IP | `ip address 192.168.1.1 255.255.255.0` |
| Enable interface | `no shutdown` |
| Save config | `write memory` |
| Configure OSPF | `router ospf 1` |
| OSPF network | `network 192.168.1.0 0.0.0.255 area 0` |
| Create VLAN | `vlan 10` |
| Trunk port | `switchport mode trunk` |
        """)

    elif net_section == "Ask engineer":
        q = st.text_input("Ask any networking question:")
        if st.button("Ask", key="net"):
            if q:
                with st.spinner("Consulting..."):
                    st.markdown(get_answer(f"Senior network engineer: {q}. Include commands and configs."))

# ════════════════════════════════════════
# ── PAGE: EMERGENCY ──
# ════════════════════════════════════════
elif page == "📞 Emergency":
    st.title("📞 Emergency Contacts")
    st.error("🚨 Life-threatening emergency — call 117 or 116 immediately.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Police", "117")
    col1.metric("Fire", "118")
    col2.metric("Ambulance", "116")
    col2.metric("Tourist Police", "+220 446 2566")
    col3.metric("Coast Guard", "+220 422 8657")
    col3.metric("Immigration", "+220 422 8631")

    st.divider()
    st.subheader("🏥 Hospitals")
    for h in [
        {"name": "Royal Victoria Teaching Hospital", "phone": "+220 422 8223", "area": "Banjul"},
        {"name": "Serekunda General Hospital", "phone": "+220 439 0765", "area": "Serekunda"},
        {"name": "MRC Gambia", "phone": "+220 449 5442", "area": "Fajara"},
        {"name": "Bansang Hospital", "phone": "+220 566 1234", "area": "CRR"},
        {"name": "Farafenni Hospital", "phone": "+220 573 1234", "area": "NBR"},
    ]:
        col_a, col_b, col_c = st.columns(3)
        col_a.write(f"**{h['name']}**")
        col_b.write(h["phone"])
        col_c.write(h["area"])

    st.divider()
    st.subheader("📱 Telecom")
    col1, col2, col3 = st.columns(3)
    col1.info("**Africell**\n111")
    col2.info("**Gamcel**\n123")
    col3.info("**QCell**\n199")

# ════════════════════════════════════════
# ── PAGE: LEGAL & LAW ──
# ════════════════════════════════════════
elif 
page == "⚖️ Legal & Law":
    st.title("⚖️ Legal & Law Guide — The Gambia")
    st.caption("Know your rights as a Gambian citizen.")

    legal_section = st.radio("Section:", [
        "Constitution",
        "Your Rights",
        "Business Law",
        "Family Law",
        "Criminal Law",
        "Land & Property",
        "AI Legal Advisor",
    ], horizontal=True)

    if legal_section == "Constitution":
        st.subheader("📜 The Gambian Constitution")
        st.info("The 1997 Constitution of the Republic of The Gambia is the supreme law of the land.")
        topics = [
            "What are the fundamental rights in the Gambia constitution?",
            "What does the Gambia constitution say about freedom of speech?",
            "What are the powers of the President of Gambia?",
            "What does the Gambia constitution say about religion?",
            "What are the rights of women in the Gambia constitution?",
            "What does the Gambia constitution say about education?",
            "What are the rights of children in Gambia?",
            "How is the National Assembly structured in Gambia?",
            "What does the Gambia constitution say about land ownership?",
            "What are the rights of arrested persons in Gambia?",
        ]
        selected = st.selectbox("Choose a constitutional topic:", topics)
        if st.button("Get explanation", type="primary"):
            with st.spinner("Reading the constitution..."):
                st.markdown(get_legal_answer(f"Based on the 1997 Gambian Constitution explain: {selected}. Give clear detailed explanation with article references where possible."))
        st.divider()
        custom_q = st.text_area("Ask your own constitutional question:")
        if st.button("Ask", key="const_ask"):
            if custom_q:
                with st.spinner("Consulting the constitution..."):
                    st.markdown(get_legal_answer(f"Based on Gambian law and constitution: {custom_q}"))

    elif legal_section == "Your Rights":
        st.subheader("🛡️ Know Your Rights in Gambia")
        rights_topics = {
            "Arrested by police": "What are my rights if I am arrested by police in Gambia?",
            "At work": "What are my employment rights as a worker in Gambia?",
            "As a tenant": "What are my rights as a tenant renting property in Gambia?",
            "As a consumer": "What are my consumer rights in Gambia?",
            "As a woman": "What are the legal rights of women in Gambia?",
            "As a child": "What are the legal rights of children in Gambia?",
            "Free speech": "What are my rights to freedom of speech in Gambia?",
            "Right to education": "What is the right to education in Gambia?",
            "Right to healthcare": "What is the right to healthcare in Gambia?",
            "Voting rights": "What are the voting rights of Gambian citizens?",
        }
        col1, col2 = st.columns(2)
        for i, (topic, query) in enumerate(rights_topics.items()):
            with col1 if i % 2 == 0 else col2:
                if st.button(f"🛡️ {topic}", key=f"rights_{i}"):
                    with st.spinner("Loading..."):
                        st.markdown(get_legal_answer(query))

    elif legal_section == "Business Law":
        st.subheader("💼 Business Law in Gambia")
        biz_topics = [
            "How do I register a business in Gambia?",
            "What taxes does a business pay in Gambia?",
            "What is the Companies Act in Gambia?",
            "How do I register an NGO in Gambia?",
            "What are the employment laws for businesses in Gambia?",
            "How do I protect my intellectual property in Gambia?",
            "What are import and export regulations in Gambia?",
            "How do I open a bank account for my business in Gambia?",
            "What licenses do I need to start a restaurant in Gambia?",
            "What are the investment laws for foreigners in Gambia?",
        ]
        selected_biz = st.selectbox("Business law topic:", biz_topics)
        if st.button("Get legal guidance", type="primary"):
            with st.spinner("Consulting business law..."):
                st.markdown(get_legal_answer(f"As a Gambian business lawyer: {selected_biz}. Give practical step by step guidance."))
        st.divider()
        st.subheader("📋 Business Registration Steps")
        st.markdown("""
**To register a business in Gambia:**
1. Choose your business structure
2. Choose a unique business name
3. Register with **GRA** for TIN
4. Register at **Registrar General's Department**
5. Get operating license from local council
6. Open a business bank account
7. Register for VAT if turnover exceeds GMD 1 million

**Key contacts:**
- Registrar General: +220 422 8181
- GRA: +220 422 7144
- GIEPA: +220 437 0765
        """)

    elif legal_section == "Family Law":
        st.subheader("👨‍👩‍👧 Family Law in Gambia")
        family_topics = [
            "How does marriage work legally in Gambia?",
            "What are the divorce laws in Gambia?",
            "What are child custody laws in Gambia?",
            "How does inheritance work in Gambia?",
            "What is the legal age of marriage in Gambia?",
            "What are the rights of widows in Gambia?",
            "How is child support determined in Gambia?",
            "What is the law on domestic violence in Gambia?",
            "How does adoption work in Gambia?",
            "What are the rights of unmarried couples in Gambia?",
        ]
        selected_family = st.selectbox("Family law topic:", family_topics)
        if st.button("Get legal guidance", type="primary", key="family_btn"):
            with st.spinner("Consulting family law..."):
                st.markdown(get_legal_answer(f"As a Gambian family lawyer explain: {selected_family}"))

    elif legal_section == "Criminal Law":
        st.subheader("⚖️ Criminal Law in Gambia")
        criminal_topics = [
            "What are the most common criminal offences in Gambia?",
            "What is the penalty for theft in Gambia?",
            "What are drug laws in Gambia?",
            "What is the legal process after arrest in Gambia?",
            "What is the role of the Gambia Police Force?",
            "What are cybercrime laws in Gambia?",
            "What is the penalty for corruption in Gambia?",
            "How does the court system work in Gambia?",
            "What are traffic laws and penalties in Gambia?",
        ]
        selected_criminal = st.selectbox("Criminal law topic:", criminal_topics)
        if st.button("Get legal guidance", type="primary", key="criminal_btn"):
            with st.spinner("Consulting criminal law..."):
                st.markdown(get_legal_answer(f"As a Gambian criminal lawyer explain: {selected_criminal}"))
        st.divider()
        st.subheader("🏛️ Court System in Gambia")
        st.markdown("""
| Court | Jurisdiction |
|-------|-------------|
| Supreme Court | Highest court — constitutional matters |
| Court of Appeal | Appeals from High Court |
| High Court | Serious criminal and civil cases |
| Magistrates Court | Minor criminal and civil cases |
| Cadi Court | Islamic personal law matters |
| District Tribunal | Local disputes and customary law |
        """)

    elif legal_section == "Land & Property":
        st.subheader("🏠 Land & Property Law in Gambia")
        land_topics = [
            "How do I buy land in Gambia legally?",
            "What is the State Lands Act in Gambia?",
            "How do I get a land certificate in Gambia?",
            "Can foreigners own land in Gambia?",
            "What are tenant rights in Gambia?",
            "How do I resolve a land dispute in Gambia?",
            "What is the process for property inheritance in Gambia?",
            "How do I verify land ownership in Gambia?",
            "What are building regulations in Gambia?",
            "How do I get planning permission in Gambia?",
        ]
        selected_land = st.selectbox("Land law topic:", land_topics)
        if st.button("Get legal guidance", type="primary", key="land_btn"):
            with st.spinner("Consulting property law..."):
                st.markdown(get_legal_answer(f"As a Gambian property lawyer explain: {selected_land}"))
        st.divider()
        st.subheader("📋 How to Buy Land in Gambia")
        st.markdown("""
**Step by step:**
1. Find land and agree on price
2. Verify ownership at **Department of Lands & Survey**
3. Hire a registered lawyer
4. Sign a sale agreement
5. Pay stamp duty at GRA
6. Register transfer at Department of Lands
7. Get your land certificate

**Key contact:**
- Department of Lands & Survey: +220 422 8400
        """)

    elif legal_section == "AI Legal Advisor":
        st.subheader("🤖 AI Legal Advisor")
        st.warning("⚠️ General legal information only. For serious matters always consult a qualified Gambian lawyer.")
        quick_legal = [
            "Can my landlord evict me without notice?",
            "What do I do if my employer does not pay me?",
            "How do I report corruption in Gambia?",
            "What happens if I am detained without charge?",
            "How do I get legal aid in Gambia?",
        ]
        for q in quick_legal:
            if st.button(f"⚖️ {q}", key=f"legal_{q}"):
                with st.spinner("Consulting..."):
                    st.markdown(get_legal_answer(f"As a Gambian legal advisor: {q}"))
        st.divider()
        legal_q = st.text_area("Your legal question:")
        if st.button("Ask legal advisor", type="primary"):
            if legal_q:
                with st.spinner("Consulting Gambian law..."):
                    st.markdown(get_legal_answer(f"As a Gambian legal advisor answer: {legal_q}. Give clear practical guidance and mention relevant laws."))
        st.divider()
        st.subheader("🏛️ Legal Aid in Gambia")
        st.markdown("""
- **Legal Aid Agency Gambia** — Free legal aid for those who cannot afford lawyers
- **Gambia Bar Association** — +220 422 8181
- **Women's Lawyers Association** — Legal help for women and children
- **Institute for Human Rights and Development** — Human rights cases
        """)