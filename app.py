import folium
from streamlit_folium import st_folium
import streamlit as st
import bcrypt
import ipaddress
from supabase import create_client
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from tavily import TavilyClient

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

@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})

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
        combined_context = f"WEB SEARCH RESULTS:\n{web_context}\n\nDOCUMENT KNOWLEDGE BASE:\n{doc_context}"
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.3
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": combined_context, "question": query})
    except:
        return "Sorry, something went wrong. Please try again."

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
        return True, "Account created successfully!"
    except Exception as e:
        return False, f"Error: {str(e)}"

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
        return False, None, f"Error: {str(e)}"

def save_message(user_id, role, content):
    try:
        supabase = get_supabase()
        supabase.table("chat_history").insert({
            "user_id": user_id,
            "role": role,
            "content": content
        }).execute()
    except:
        pass

def load_history(user_id):
    try:
        supabase = get_supabase()
        result = supabase.table("chat_history")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at")\
            .limit(50)\
            .execute()
        return result.data or []
    except:
        return []

def clear_history(user_id):
    try:
        supabase = get_supabase()
        supabase.table("chat_history").delete().eq("user_id", user_id).execute()
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

# --- PAGE CONFIG ---
st.set_page_config(page_title="GambiaGPT", page_icon="🇬🇲", layout="centered")
st.title("🇬🇲 GambiaGPT")
st.caption("Your AI guide to The Gambia — and your cybersecurity tutor")

# --- AUTH STATE ---
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: LOGIN / REGISTER ---
with st.sidebar:
    st.header("👤 Account")
    if st.session_state.user:
        st.success(f"Logged in as:\n{st.session_state.user['email']}")
        if st.button("Load my chat history"):
            history = load_history(st.session_state.user["id"])
            st.session_state.messages = [{"role": h["role"], "content": h["content"]} for h in history]
            st.rerun()
        if st.button("Clear chat history"):
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
                        st.error("Password must be at least 6 characters.")
                    else:
                        success, msg = register_user(email, password)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Ask GambiaGPT", "🔐 Cybersecurity Lab", "🌐 Networking Lab", "🗺️ Gambia Map", "📞 Emergency Contacts"])
# ── TAB 1: CHAT ──
with tab1:
    st.info("💬 Ask in English, Mandinka, Wolof, Jola or Fula — powered by live web search.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask anything about Gambia or cybersecurity..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Searching and thinking..."):
                answer = get_answer(query)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if st.session_state.user:
            save_message(st.session_state.user["id"], "user", query)
            save_message(st.session_state.user["id"], "assistant", answer)

# ── TAB 2: CYBERSECURITY LAB ──
with tab2:
    st.subheader("🔐 Cybersecurity Learning Lab")
    topic = st.selectbox("Choose a topic to study:", [
        "Pick a topic...",
        "CEH — Certified Ethical Hacker overview",
        "CompTIA Security+ study guide",
        "Kali Linux top 10 tools",
        "How to do a penetration test",
        "Types of cyberattacks explained",
        "Firewalls and how they work",
        "VPN explained",
        "Password attacks explained",
        "Social engineering and phishing",
        "How to start a cybersecurity career in Gambia",
        "CTF — Capture The Flag beginner guide",
        "Python for cybersecurity automation",
        "OWASP Top 10 web vulnerabilities",
        "Network scanning with Nmap",
        "Wireshark packet analysis basics",
    ])
    if topic != "Pick a topic...":
        if st.button("Learn this topic", type="primary"):
            with st.spinner(f"Loading {topic}..."):
                answer = get_answer(f"Teach me about: {topic}. Give a detailed explanation with examples, commands, and practical tips.")
                st.markdown(answer)
    st.divider()
    cyber_q = st.text_input("Ask any cybersecurity question:")
    if st.button("Ask tutor", key="cyber_btn"):
        if cyber_q:
            with st.spinner("Thinking..."):
                st.markdown(get_answer(f"As a cybersecurity expert: {cyber_q}"))
    st.divider()
    st.subheader("🎯 Cybersecurity Roadmap for Gambians")
    st.markdown("""
**Beginner:** CompTIA ITF+ → A+ → Network+ → Security+

**Intermediate:** CEH → eJPT → OSCP

**Free resources:**
- [TryHackMe](https://tryhackme.com) — best for beginners
- [HackTheBox](https://hackthebox.com) — intermediate labs
- [Cybrary](https://cybrary.it) — free courses
- [Professor Messer](https://professormesser.com) — CompTIA videos
    """)

# ── TAB 3: NETWORKING LAB ──
with tab3:
    st.subheader("🌐 Networking & Routing Lab")
    net_topic = st.selectbox("Choose a networking topic:", [
        "Pick a topic...",
        "OSI model — all 7 layers explained",
        "TCP/IP model vs OSI model",
        "How routing works — step by step",
        "OSPF configuration on Cisco routers",
        "BGP — Border Gateway Protocol explained",
        "EIGRP configuration guide",
        "VLANs — setup and trunking",
        "Spanning Tree Protocol STP explained",
        "NAT and PAT — how they work",
        "DHCP — how IP addresses are assigned",
        "DNS — how domain names resolve",
        "Access Control Lists ACLs on Cisco",
        "How switches work — MAC address table",
        "IPv6 subnetting and configuration",
        "GRE tunnels and VPN configuration",
        "CCNA exam study guide",
    ])
    if net_topic != "Pick a topic...":
        if st.button("Learn this topic", type="primary", key="net_btn"):
            with st.spinner(f"Loading {net_topic}..."):
                answer = get_answer(f"As a CCNA instructor, teach me: {net_topic}. Include Cisco IOS commands and real configs.")
                st.markdown(answer)
    st.divider()
    st.subheader("🧮 Subnet Calculator")
    col1, col2 = st.columns(2)
    with col1:
        ip_input = st.text_input("IP Address", value="192.168.1.0")
    with col2:
        prefix_input = st.slider("Prefix length", min_value=8, max_value=30, value=24)
    if st.button("Calculate subnet", type="primary"):
        result = subnet_calculator(ip_input, prefix_input)
        if result:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Network address", result["network"])
                st.metric("Subnet mask", result["netmask"])
                st.metric("First host", result["first_host"])
                st.metric("IP class", result["ip_class"])
            with col_b:
                st.metric("Broadcast address", result["broadcast"])
                st.metric("Wildcard mask", result["wildcard"])
                st.metric("Last host", result["last_host"])
                st.metric("Usable hosts", f"{result['hosts']:,}")
        else:
            st.error("Invalid IP address. Please check and try again.")
    st.divider()
    st.subheader("💻 Cisco IOS Quick Reference")
    st.markdown("""
| Task | Command |
|------|---------|
| Enter privileged mode | `enable` |
| Enter config mode | `configure terminal` |
| Set hostname | `hostname R1` |
| Show IP interfaces | `show ip interface brief` |
| Show routing table | `show ip route` |
| Configure interface | `interface gi0/0` |
| Set IP address | `ip address 192.168.1.1 255.255.255.0` |
| Enable interface | `no shutdown` |
| Save config | `write memory` |
| Configure OSPF | `router ospf 1` |
| Set OSPF network | `network 192.168.1.0 0.0.0.255 area 0` |
| Configure VLAN | `vlan 10` |
| Set trunk port | `switchport mode trunk` |
    """)
    net_q = st.text_input("Ask any networking question:")
    if st.button("Ask engineer", key="net_ask_btn"):
        if net_q:
            with st.spinner("Consulting the network engineer..."):
                st.markdown(get_answer(f"As a senior network engineer: {net_q}. Include commands and configs."))
                # ── TAB 4: INTERACTIVE MAP ──
with tab4:
    st.subheader("🗺️ Interactive Map of The Gambia")
    st.caption("Explore tourist spots, hospitals, universities and more.")

    # Filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        map_filter = st.multiselect(
            "Show on map:",
            ["Tourist spots", "Hospitals", "Universities", "Hotels", "Markets", "Embassies"],
            default=["Tourist spots"]
        )
    with col2:
        map_style = st.selectbox("Map style:", ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"])

    # Create base map centered on Gambia
    m = folium.Map(
        location=[13.4549, -15.3100],
        zoom_start=8,
        tiles=map_style
    )

    # --- TOURIST SPOTS ---
    tourist_spots = [
        {"name": "Kachikally Crocodile Pool", "lat": 13.4441, "lon": -16.6774, "desc": "Sacred crocodile pool in Bakau. One of Gambia's most visited attractions.", "region": "West Coast"},
        {"name": "Abuko Nature Reserve", "lat": 13.3833, "lon": -16.6500, "desc": "Gambia's first nature reserve. Home to monkeys, birds, and reptiles.", "region": "West Coast"},
        {"name": "Janjanbureh Island", "lat": 13.5500, "lon": -14.7667, "desc": "Historic island town. Former colonial capital, rich in history.", "region": "Central River"},
        {"name": "Wassu Stone Circles", "lat": 13.6833, "lon": -14.9167, "desc": "UNESCO World Heritage Site. Ancient stone circles dating back 1,000 years.", "region": "Central River"},
        {"name": "River Gambia National Park", "lat": 13.5833, "lon": -14.9000, "desc": "Home to chimpanzees and hippos along the River Gambia.", "region": "Central River"},
        {"name": "Bijilo Forest Park", "lat": 13.4167, "lon": -16.7167, "desc": "Coastal forest park with green vervet monkeys and diverse birdlife.", "region": "West Coast"},
        {"name": "Tanji Bird Reserve", "lat": 13.3667, "lon": -16.7167, "desc": "Important bird watching site on the Atlantic coast.", "region": "West Coast"},
        {"name": "Kartong Beach", "lat": 13.0833, "lon": -16.7500, "desc": "Unspoiled beach at the southern tip of Gambia. Peaceful and scenic.", "region": "West Coast"},
        {"name": "Banjul Albert Market", "lat": 13.4531, "lon": -16.5731, "desc": "Gambia's biggest market. Everything from fabrics to spices and crafts.", "region": "Banjul"},
        {"name": "Arch 22", "lat": 13.4533, "lon": -16.5756, "desc": "Iconic monument in Banjul. Great views of the city from the top.", "region": "Banjul"},
        {"name": "Serrekunda Market", "lat": 13.4386, "lon": -16.6775, "desc": "Largest market in Gambia. Vibrant and full of local culture.", "region": "West Coast"},
        {"name": "James Island", "lat": 13.4833, "lon": -16.5333, "desc": "UNESCO World Heritage Site. Former slave trade fort in the River Gambia.", "region": "West Coast"},
    ]

    # --- HOSPITALS ---
    hospitals = [
        {"name": "Royal Victoria Teaching Hospital", "lat": 13.4544, "lon": -16.5786, "desc": "Gambia's main national hospital in Banjul. Largest medical facility.", "region": "Banjul"},
        {"name": "Edward Francis Small Teaching Hospital", "lat": 13.4544, "lon": -16.5786, "desc": "Main referral hospital serving greater Banjul area.", "region": "Banjul"},
        {"name": "Serekunda General Hospital", "lat": 13.4386, "lon": -16.6775, "desc": "Major hospital serving the greater Serekunda area.", "region": "West Coast"},
        {"name": "Bansang Hospital", "lat": 13.4667, "lon": -14.6500, "desc": "Main hospital for Central River Region.", "region": "Central River"},
        {"name": "Farafenni Hospital", "lat": 13.5667, "lon": -15.6000, "desc": "Main hospital for North Bank Region.", "region": "North Bank"},
        {"name": "Kanifing Hospital", "lat": 13.4500, "lon": -16.6667, "desc": "Hospital serving Kanifing Municipal area.", "region": "West Coast"},
        {"name": "MRC Gambia", "lat": 13.4167, "lon": -16.6667, "desc": "Medical Research Council — world-class research hospital.", "region": "West Coast"},
    ]

    # --- UNIVERSITIES ---
    universities = [
        {"name": "University of The Gambia", "lat": 13.4167, "lon": -16.6500, "desc": "Gambia's national university. Offers degrees in medicine, law, engineering and more.", "region": "West Coast"},
        {"name": "GTTI — Gambia Technical Training Institute", "lat": 13.4500, "lon": -16.6500, "desc": "Technical and vocational training institute.", "region": "West Coast"},
        {"name": "Gambia College", "lat": 13.3833, "lon": -16.6833, "desc": "Teacher training and nursing college.", "region": "West Coast"},
        {"name": "American International University", "lat": 13.4500, "lon": -16.6400, "desc": "Private international university in Kanifing.", "region": "West Coast"},
        {"name": "Management Development Institute", "lat": 13.4400, "lon": -16.6600, "desc": "Business and management studies institute.", "region": "West Coast"},
    ]

    # --- HOTELS ---
    hotels = [
        {"name": "Coco Ocean Resort", "lat": 13.4167, "lon": -16.7333, "desc": "5-star luxury resort on the Atlantic coast.", "region": "West Coast"},
        {"name": "Sunset Beach Hotel", "lat": 13.4167, "lon": -16.7167, "desc": "Popular beachfront hotel in Kololi.", "region": "West Coast"},
        {"name": "Kairaba Beach Hotel", "lat": 13.4167, "lon": -16.7200, "desc": "One of Gambia's top beach hotels.", "region": "West Coast"},
        {"name": "Laico Atlantic Hotel", "lat": 13.4533, "lon": -16.5867, "desc": "Business hotel in central Banjul.", "region": "Banjul"},
        {"name": "Ngala Lodge", "lat": 13.4833, "lon": -16.6833, "desc": "Boutique eco-lodge in Bakau.", "region": "West Coast"},
        {"name": "Mandina Lodges", "lat": 13.5000, "lon": -15.8000, "desc": "Stunning eco-lodge on the River Gambia.", "region": "Central River"},
    ]

    # --- MARKETS ---
    markets = [
        {"name": "Albert Market Banjul", "lat": 13.4531, "lon": -16.5731, "desc": "Gambia's oldest and most famous market.", "region": "Banjul"},
        {"name": "Serrekunda Market", "lat": 13.4386, "lon": -16.6775, "desc": "Largest market in Gambia.", "region": "West Coast"},
        {"name": "Brikama Market", "lat": 13.2667, "lon": -16.6500, "desc": "Known for wood carvings and crafts.", "region": "West Coast"},
        {"name": "Tanji Fish Market", "lat": 13.3667, "lon": -16.7167, "desc": "Busy fishing village market on the coast.", "region": "West Coast"},
        {"name": "Bakau Market", "lat": 13.4667, "lon": -16.6833, "desc": "Local market in Bakau town.", "region": "West Coast"},
    ]

    # --- EMBASSIES ---
    embassies = [
        {"name": "US Embassy Banjul", "lat": 13.4533, "lon": -16.5800, "desc": "United States Embassy in Banjul.", "region": "Banjul"},
        {"name": "UK High Commission", "lat": 13.4500, "lon": -16.5750, "desc": "British High Commission in Banjul.", "region": "Banjul"},
        {"name": "EU Delegation Gambia", "lat": 13.4520, "lon": -16.5780, "desc": "European Union Delegation in Banjul.", "region": "Banjul"},
        {"name": "Senegal Embassy", "lat": 13.4510, "lon": -16.5760, "desc": "Embassy of Senegal in Banjul.", "region": "Banjul"},
        {"name": "China Embassy", "lat": 13.4490, "lon": -16.5790, "desc": "Embassy of China in Banjul.", "region": "Banjul"},
    ]

    # Color and icon mapping
    category_config = {
        "Tourist spots": {"data": tourist_spots, "color": "green", "icon": "star"},
        "Hospitals": {"data": hospitals, "color": "red", "icon": "plus"},
        "Universities": {"data": universities, "color": "blue", "icon": "graduation-cap"},
        "Hotels": {"data": hotels, "color": "purple", "icon": "home"},
        "Markets": {"data": markets, "color": "orange", "icon": "shopping-cart"},
        "Embassies": {"data": embassies, "color": "darkblue", "icon": "flag"},
    }

    # Add markers for selected categories
    for category in map_filter:
        if category in category_config:
            config = category_config[category]
            for place in config["data"]:
                folium.Marker(
                    location=[place["lat"], place["lon"]],
                    popup=folium.Popup(
                        f"<b>{place['name']}</b><br>{place['desc']}<br><i>{place['region']}</i>",
                        max_width=250
                    ),
                    tooltip=place["name"],
                    icon=folium.Icon(
                        color=config["color"],
                        icon=config["icon"],
                        prefix="fa"
                    )
                ).add_to(m)

    # Add river Gambia line
    folium.PolyLine(
        locations=[
            [13.4667, -16.5833],
            [13.4833, -16.3000],
            [13.5000, -15.8000],
            [13.5500, -15.3000],
            [13.5500, -14.9000],
            [13.5500, -14.6500],
        ],
        color="blue",
        weight=3,
        opacity=0.6,
        tooltip="River Gambia"
    ).add_to(m)

    # Display map
    map_data = st_folium(m, width=700, height=500)

    # Show clicked location info
    if map_data["last_object_clicked_popup"]:
        st.info(f"Selected: {map_data['last_object_clicked_popup']}")

    st.divider()

    # Stats
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Tourist spots", len(tourist_spots))
    col_b.metric("Hospitals", len(hospitals))
    col_c.metric("Universities", len(universities))
    col_d.metric("Hotels", len(hotels))

    st.caption("Click any marker on the map to see details. Use the filter above to show different categories.")
    # ── TAB 5: EMERGENCY CONTACTS ──
with tab5:
    st.subheader("📞 Emergency Contacts — The Gambia")
    st.error("🚨 If this is a life-threatening emergency call 117 or 116 immediately.")

    st.divider()

    # Emergency services
    st.subheader("🚨 Emergency Services")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Police Emergency", "117")
        st.metric("Fire Service", "118")
    with col2:
        st.metric("Ambulance", "116")
        st.metric("Tourist Police", "+220 446 2566")
    with col3:
        st.metric("Coast Guard", "+220 422 8657")
        st.metric("Immigration", "+220 422 8631")

    st.divider()

    # Hospitals
    st.subheader("🏥 Major Hospitals")
    hospitals = [
        {"name": "Royal Victoria Teaching Hospital", "phone": "+220 422 8223", "location": "Banjul", "notes": "Main national hospital"},
        {"name": "Serekunda General Hospital", "phone": "+220 439 0765", "location": "Serekunda", "notes": "Largest hospital outside Banjul"},
        {"name": "Bansang Hospital", "phone": "+220 566 1234", "location": "Bansang, CRR", "notes": "Central River Region"},
        {"name": "Farafenni Hospital", "phone": "+220 573 1234", "location": "Farafenni, NBR", "notes": "North Bank Region"},
        {"name": "MRC Gambia", "phone": "+220 449 5442", "location": "Fajara", "notes": "Medical Research Centre"},
        {"name": "Kanifing Hospital", "phone": "+220 439 2620", "location": "Kanifing", "notes": "KMC area hospital"},
    ]
    for h in hospitals:
        with st.expander(f"🏥 {h['name']} — {h['location']}"):
            col_a, col_b = st.columns(2)
            col_a.write(f"📞 **Phone:** {h['phone']}")
            col_b.write(f"📍 **Location:** {h['location']}")
            st.write(f"ℹ️ {h['notes']}")

    st.divider()

    # Government
    st.subheader("🏛️ Government Contacts")
    gov_contacts = [
        {"name": "State House", "phone": "+220 422 2745", "dept": "Office of the President"},
        {"name": "Ministry of Health", "phone": "+220 422 8428", "dept": "Health Services"},
        {"name": "Ministry of Education", "phone": "+220 422 8833", "dept": "Education"},
        {"name": "Ministry of Justice", "phone": "+220 422 8181", "dept": "Legal Affairs"},
        {"name": "Ministry of Finance", "phone": "+220 422 7571", "dept": "Finance"},
        {"name": "Ministry of Foreign Affairs", "phone": "+220 422 9400", "dept": "Foreign Affairs"},
        {"name": "Gambia Revenue Authority", "phone": "+220 422 7144", "dept": "Taxes & Revenue"},
        {"name": "NAWEC", "phone": "+220 422 5544", "dept": "Water & Electricity"},
        {"name": "Gambia Ports Authority", "phone": "+220 422 7266", "dept": "Ports & Shipping"},
    ]
    for g in gov_contacts:
        col_a, col_b, col_c = st.columns(3)
        col_a.write(f"**{g['name']}**")
        col_b.write(g['phone'])
        col_c.write(g['dept'])

    st.divider()

    # Embassies
    st.subheader("🌍 Embassies & High Commissions in Banjul")
    embassies = [
        {"country": "United States", "phone": "+220 439 2856", "address": "Kairaba Avenue, Fajara"},
        {"country": "United Kingdom", "phone": "+220 449 5133", "address": "48 Atlantic Road, Fajara"},
        {"country": "Senegal", "phone": "+220 422 7469", "address": "10 Nelson Mandela Street, Banjul"},
        {"country": "China", "phone": "+220 422 8839", "address": "Kairaba Avenue, Fajara"},
        {"country": "European Union", "phone": "+220 449 5018", "address": "48 Kairaba Avenue"},
        {"country": "Nigeria", "phone": "+220 422 8483", "address": "31 Liberation Avenue, Banjul"},
        {"country": "Ghana", "phone": "+220 422 7870", "address": "2 Clarkson Street, Banjul"},
        {"country": "Sierra Leone", "phone": "+220 422 9250", "address": "67 Hagan Street, Banjul"},
    ]
    for e in embassies:
        with st.expander(f"🌍 {e['country']}"):
            st.write(f"📞 **Phone:** {e['phone']}")
            st.write(f"📍 **Address:** {e['address']}")

    st.divider()

    # Telecom operators
    st.subheader("📱 Telecom Operators")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Africell**\nCustomer care: 111\nWebsite: africell.gm")
    with col2:
        st.info("**Gamcel**\nCustomer care: 123\nWebsite: gamcel.gm")
    with col3:
        st.info("**QCell**\nCustomer care: 199\nWebsite: qcell.gm")

    st.divider()

    # Useful links
    st.subheader("🔗 Useful Links")
    st.markdown("""
- [Gambia Government Portal](https://www.gov.gm)
- [Gambia Tourism Board](https://www.visitthegambia.gm)
- [NAWEC — Power & Water](https://www.nawec.gm)
- [Gambia Revenue Authority](https://www.gra.gm)
- [University of The Gambia](https://www.utm.edu.gm)
- [Gambia Police Force](https://www.gambiapolice.gm)
    """)

    st.caption("📌 Always verify contact numbers locally as they may change. In emergency always call 117 or 116.")