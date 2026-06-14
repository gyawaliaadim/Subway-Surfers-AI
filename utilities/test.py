label=1
to_flip=True

if to_flip:
    flip_map = {
        1: 3,  # left → right
        3: 1   # right → left
    }  

    label = flip_map.get(label)  
    print(label)   

label=1
to_flip=False

if to_flip:
    flip_map = {
        1: 3,  # left → right
        3: 1   # right → left
    }  

    label = flip_map.get(label)  
    print(label)  
print(label) 
