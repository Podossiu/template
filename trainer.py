import yaml

def main():
    with open('./app/resnet_18.yml','r') as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

if __name__ == '__main__':
    main()
    
