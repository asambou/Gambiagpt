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
tab1, tab2, tab3 = st.tabs(["💬 Ask GambiaGPT", "🔐 Cybersecurity Lab", "🌐 Networking Lab"])

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