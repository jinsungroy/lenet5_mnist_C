#include "read_mnist.h"
#include <iostream>
#include <fstream>
using namespace std;

int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

void read_mnist_images(float* mnistData, string full_path)
{
    ifstream file(full_path.c_str(), ios::binary);

    if(file.is_open())
    {
        int magic_number = 0, n_rows = 0, n_cols = 0, number_of_images;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");
        file.read((char *)&number_of_images, sizeof(number_of_images));
        file.read((char *)&n_rows, sizeof(n_rows));
        file.read((char *)&n_cols, sizeof(n_cols));
        number_of_images = reverseInt(number_of_images);
        n_rows = reverseInt(n_rows);
        n_cols = reverseInt(n_cols);

        for(int num = 0; num < number_of_images; num++)
        {
        	for(int i=0; i<32; i++)
        	{
        		for(int j=0; j<32; j++)
        		{
        			float temp = 0;
        			if(i<2 || j<2 || i>29 || j>29)
        			{
        				temp = 0;
        			}
        			else
        			{
        				unsigned char value = 0;
        				file.read((char *)&value, sizeof(value));
        				temp = float(value) / 255;
        			}
        			mnistData[num*1024+i*32+j] = temp;
        		}
        	}

        }
        cout << "Read MNIST Data done" << endl;
    }
    else
    {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}


void read_mnist_labels(int* mnistLabel, string full_path)
{
    ifstream file(full_path.c_str(), ios::binary);

    if(file.is_open()) 
	{
    	int magic_number = 0, number_of_labels = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        for(int i = 0; i < number_of_labels; i++)
        {
        	unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            mnistLabel[i] = (int)temp;
        }
    }
    else
    {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}
