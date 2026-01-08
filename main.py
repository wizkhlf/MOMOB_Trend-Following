from app import app
import os
if __name__ == "__main__":
    url = "http://127.0.0.1:5002"

    print("\n============================================")
    print(f" Cliquer pour ouvrir l'application : {url}")
    print("============================================\n")

    

    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port)


