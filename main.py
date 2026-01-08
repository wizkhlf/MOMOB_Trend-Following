from app import app

if __name__ == "__main__":
    url = "http://127.0.0.1:5002"

    print("\n============================================")
    print(f" Cliquer pour ouvrir l'application : {url}")
    print("============================================\n")

    # On lance le serveur Flask
    app.run(debug=True, host="0.0.0.0", port=5002)
