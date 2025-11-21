from Crypto.PublicKey import RSA

def generate_keys(name: str):
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()

    with open(f"./keys/{name}_private.pem", "wb") as f:
        f.write(private_key)
    with open(f"./keys/{name}_public.pem", "wb") as f:
        f.write(public_key)

    print(f"Generated keys for {name}")

def main():
    participants = ["worker0", "worker1", "worker2","worker3","worker4","worker5", "worker6","worker7", "coordinator"]
    
    for name in participants:
        generate_keys(name)

if __name__ == "__main__":
    main()
