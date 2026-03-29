🤖 Alexa AI Assistant — Gemini + Brave
Skill de Alexa que actúa como asistente inteligente con dos modos de respuesta: internet en tiempo real (Brave) o conocimiento general (Gemini).

✨ Características
🔍 Modo internet — busca información actual usando la API de Brave
🧠 Modo conocimiento general — responde con Gemini sin necesidad de conexión
💾 Historial persistente — guarda conversaciones en DynamoDB con embeddings vectoriales
⚡ Guardado asíncrono — no bloquea la respuesta de Alexa
🔄 Rotación de claves API — distribuye el uso entre múltiples claves automáticamente
🗣️ Optimizado para voz — respuestas sin Markdown, listas ni formatos
🛠️ Tecnologías
Servicio	Uso
AWS Lambda	Ejecución del código
AWS DynamoDB	Historial de conversaciones
Gemini API	Respuestas de conocimiento general + embeddings
Brave Search API	Búsquedas en tiempo real
Alexa Skills Kit	Interfaz de voz
📁 Estructura del proyecto
alexa-gemini-skill/ ├── lambda/ │ ├── lambda_function.py │ └── requirements.txt ├── skill/ │ └── interactionModel.json ├── .env.example ├── .gitignore └── README.md

⚙️ Configuración
1. Variables de entorno en Lambda
Copia .env.example y rellena con tus claves:

GEMINI_API_KEY_1=tu_clave GEMINI_API_KEY_2=tu_clave GEMINI_API_KEY_3=tu_clave GEMINI_API_KEY_4=tu_clave BRAVE_API_KEY_1=tu_clave BRAVE_API_KEY_2=tu_clave DYNAMODB_TABLE=laia-historial AWS_REGION=eu-west-1

2. DynamoDB
Crea una tabla con:

Nombre: laia-historial (o el que definas en DYNAMODB_TABLE)
Partition key: user_id (String)
Sort key: timestamp (String)
3. Instalar dependencias y empaquetar
pip install -r requirements.txt -t ./package
cp lambda/lambda_function.py ./package/
cd package
zip -r ../skill.zip .
Sube skill.zip a tu función Lambda.

4. Modelo de interacción de Alexa
En la Alexa Developer Console:

Crea una nueva skill en español
Ve a JSON Editor
Pega el contenido de skill/interactionModel.json
Haz clic en Save Model → Build Model
5. Conectar Lambda con Alexa
Copia el ARN de tu función Lambda y pégalo en el endpoint de la skill.

🗣️ Cómo usarlo
Usuario: "Alexa, abre [nombre de tu skill]" Alexa: "Hola, soy tu asistente. ¿Vas a hacer preguntas sobre información actual?" Usuario: "Sí" → activa modo Brave (internet) "No" → activa modo Gemini (conocimiento general)

Usuario: "Dime quién ganó el último Gran Premio de Fórmula 1" Usuario: "Explícame cómo funciona un motor de hidrógeno" Usuario: "Nada más" / "Adiós" → cierra la sesión

📦 Dependencias
ask-sdk-core>=1.19.0 requests>=2.28.0 boto3>=1.26.0

⚠️ boto3 ya viene preinstalado en Lambda. Solo es necesario para desarrollo local.

🔑 Obtener las claves API
Gemini: aistudio.google.com
Brave Search: brave.com/search/api
📄 Licencia
MIT — libre para usar, modificar y distribuir.

Solo tienes que sustituir [nombre de tu skill] por el nombre real de invocación que hayas configurado en Alexa.
