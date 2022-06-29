device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"We use: {device}")