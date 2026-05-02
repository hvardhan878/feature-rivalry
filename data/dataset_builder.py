import json
import os
import random
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from config import (
    MAX_NEW_TOKENS,
    N_SAMPLES_FOR_ENTROPY,
    TEMPERATURE_SAMPLING,
)


# Fallback: 600+ simple factual Q&A pairs (used if Natural Questions fails or returns too few)
_FALLBACK_UNAMBIGUOUS_QA = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"question": "What planet is closest to the Sun?", "answer": "Mercury"},
    {"question": "How many sides does a hexagon have?", "answer": "Six"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific"},
    {"question": "In what year did World War II end?", "answer": "1945"},
    {"question": "What is the square root of 144?", "answer": "12"},
    {"question": "Who was the first US president?", "answer": "George Washington"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "How many continents are there?", "answer": "Seven"},
    {"question": "What is the boiling point of water in Celsius?", "answer": "100"},
    {"question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
    {"question": "What is the smallest prime number?", "answer": "Two"},
    {"question": "What planet is known as the Red Planet?", "answer": "Mars"},
    {"question": "How many days are in a leap year?", "answer": "366"},
    {"question": "What is the capital of Germany?", "answer": "Berlin"},
    {"question": "Who invented the light bulb?", "answer": "Thomas Edison"},
    {"question": "What is 7 times 8?", "answer": "56"},
    {"question": "What is the longest river in the world?", "answer": "Nile"},
    {"question": "What is the capital of Australia?", "answer": "Canberra"},
    {"question": "How many sides does a triangle have?", "answer": "Three"},
    {"question": "What is the chemical symbol for iron?", "answer": "Fe"},
    {"question": "Who wrote Pride and Prejudice?", "answer": "Jane Austen"},
    {"question": "What year did the Titanic sink?", "answer": "1912"},
    {"question": "What is the freezing point of water in Fahrenheit?", "answer": "32"},
    {"question": "What is the capital of Italy?", "answer": "Rome"},
    {"question": "How many hours are in a day?", "answer": "24"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "Who painted Starry Night?", "answer": "Vincent van Gogh"},
    {"question": "What is 15 plus 27?", "answer": "42"},
    {"question": "What is the capital of Spain?", "answer": "Madrid"},
    {"question": "How many legs does a spider have?", "answer": "Eight"},
    {"question": "What is the chemical symbol for oxygen?", "answer": "O"},
    {"question": "Who wrote 1984?", "answer": "George Orwell"},
    {"question": "What is the speed of light in vacuum?", "answer": "299792458"},
    {"question": "What is the capital of Canada?", "answer": "Ottawa"},
    {"question": "How many minutes are in an hour?", "answer": "60"},
    {"question": "What is the smallest country in the world?", "answer": "Vatican City"},
    {"question": "Who discovered gravity?", "answer": "Isaac Newton"},
    {"question": "What is 100 divided by 4?", "answer": "25"},
    {"question": "What is the capital of Brazil?", "answer": "Brasília"},
    {"question": "How many bones are in the adult human body?", "answer": "206"},
    {"question": "What is the chemical symbol for carbon?", "answer": "C"},
    {"question": "Who wrote The Great Gatsby?", "answer": "F. Scott Fitzgerald"},
    {"question": "What is the tallest mountain on Earth?", "answer": "Everest"},
    {"question": "What is the capital of India?", "answer": "New Delhi"},
    {"question": "How many weeks are in a year?", "answer": "52"},
    {"question": "What planet has the most moons?", "answer": "Saturn"},
    {"question": "Who wrote Macbeth?", "answer": "William Shakespeare"},
    {"question": "What is 12 squared?", "answer": "144"},
    {"question": "What is the capital of China?", "answer": "Beijing"},
    {"question": "How many teeth do adult humans have?", "answer": "32"},
    {"question": "What is the chemical symbol for silver?", "answer": "Ag"},
    {"question": "Who wrote Moby Dick?", "answer": "Herman Melville"},
    {"question": "What is the largest desert in the world?", "answer": "Antarctica"},
    {"question": "What is the capital of Russia?", "answer": "Moscow"},
    {"question": "How many seconds are in a minute?", "answer": "60"},
    {"question": "What is the only even prime number?", "answer": "Two"},
    {"question": "Who painted The Persistence of Memory?", "answer": "Salvador Dalí"},
    {"question": "What is 25 times 4?", "answer": "100"},
    {"question": "What is the capital of Egypt?", "answer": "Cairo"},
    {"question": "How many chromosomes do humans have?", "answer": "46"},
    {"question": "What is the chemical symbol for sodium?", "answer": "Na"},
    {"question": "Who wrote To Kill a Mockingbird?", "answer": "Harper Lee"},
    {"question": "What is the largest mammal on Earth?", "answer": "Blue whale"},
    {"question": "What is the capital of South Africa?", "answer": "Pretoria"},
    {"question": "How many faces does a cube have?", "answer": "Six"},
    {"question": "What planet has visible rings?", "answer": "Saturn"},
    {"question": "Who developed the theory of relativity?", "answer": "Albert Einstein"},
    {"question": "What is the square root of 64?", "answer": "8"},
    {"question": "What is the capital of Mexico?", "answer": "Mexico City"},
    {"question": "How many keys are on a standard piano?", "answer": "88"},
    {"question": "What is the chemical symbol for hydrogen?", "answer": "H"},
    {"question": "Who wrote The Catcher in the Rye?", "answer": "J.D. Salinger"},
    {"question": "What is the deepest ocean trench?", "answer": "Mariana"},
    {"question": "What is the capital of Argentina?", "answer": "Buenos Aires"},
    {"question": "How many sides does an octagon have?", "answer": "Eight"},
    {"question": "What is the atomic number of helium?", "answer": "2"},
    {"question": "Who wrote Crime and Punishment?", "answer": "Fyodor Dostoevsky"},
    {"question": "What is 9 times 9?", "answer": "81"},
    {"question": "What is the capital of South Korea?", "answer": "Seoul"},
    {"question": "How many stripes are on the US flag?", "answer": "13"},
    {"question": "What is the chemical symbol for nitrogen?", "answer": "N"},
    {"question": "Who wrote War and Peace?", "answer": "Leo Tolstoy"},
    {"question": "What is the hottest planet in our solar system?", "answer": "Venus"},
    {"question": "What is the capital of Indonesia?", "answer": "Jakarta"},
    {"question": "How many states are in the United States?", "answer": "50"},
    {"question": "What is pi approximately equal to?", "answer": "3.14"},
    {"question": "Who painted The Scream?", "answer": "Edvard Munch"},
    {"question": "What is 50 percent of 200?", "answer": "100"},
    {"question": "What is the capital of Turkey?", "answer": "Ankara"},
    {"question": "How many legs does an insect have?", "answer": "Six"},
    {"question": "What is the chemical symbol for calcium?", "answer": "Ca"},
    {"question": "Who wrote The Odyssey?", "answer": "Homer"},
    {"question": "What is the largest bird in the world?", "answer": "Ostrich"},
    {"question": "What is the capital of Poland?", "answer": "Warsaw"},
    {"question": "How many sides does a pentagon have?", "answer": "Five"},
    {"question": "What is absolute zero in Celsius?", "answer": "-273.15"},
    {"question": "Who wrote Frankenstein?", "answer": "Mary Shelley"},
    {"question": "What is 2 to the power of 10?", "answer": "1024"},
    {"question": "What is the capital of Thailand?", "answer": "Bangkok"},
    {"question": "How many bones are in the human spine?", "answer": "33"},
    {"question": "What is the chemical symbol for potassium?", "answer": "K"},
    {"question": "Who wrote Dracula?", "answer": "Bram Stoker"},
    {"question": "What is the smallest planet in our solar system?", "answer": "Mercury"},
    {"question": "What is the capital of Vietnam?", "answer": "Hanoi"},
    {"question": "How many letters are in the English alphabet?", "answer": "26"},
    {"question": "What is the speed of sound in air?", "answer": "343"},
    {"question": "Who painted the Sistine Chapel ceiling?", "answer": "Michelangelo"},
    {"question": "What is 17 times 3?", "answer": "51"},
    {"question": "What is the capital of Greece?", "answer": "Athens"},
    {"question": "How many days are in October?", "answer": "31"},
    {"question": "What is the chemical symbol for copper?", "answer": "Cu"},
    {"question": "Who wrote The Hobbit?", "answer": "J.R.R. Tolkien"},
    {"question": "What is the largest internal organ?", "answer": "Liver"},
    {"question": "What is the capital of Portugal?", "answer": "Lisbon"},
    {"question": "How many faces does a tetrahedron have?", "answer": "Four"},
    {"question": "What is Earth's circumference in km?", "answer": "40075"},
    {"question": "Who wrote Anna Karenina?", "answer": "Leo Tolstoy"},
    {"question": "What is 144 divided by 12?", "answer": "12"},
    {"question": "What is the capital of Sweden?", "answer": "Stockholm"},
    {"question": "How many chambers does the human heart have?", "answer": "Four"},
    {"question": "What is the chemical symbol for lead?", "answer": "Pb"},
    {"question": "Who wrote The Divine Comedy?", "answer": "Dante"},
    {"question": "What is the longest bone in the human body?", "answer": "Femur"},
    {"question": "What is the capital of Norway?", "answer": "Oslo"},
    {"question": "How many players are on a soccer team?", "answer": "11"},
    {"question": "What is the atomic number of carbon?", "answer": "6"},
    {"question": "Who painted Girl with a Pearl Earring?", "answer": "Vermeer"},
    {"question": "What is 20 percent of 80?", "answer": "16"},
    {"question": "What is the capital of Finland?", "answer": "Helsinki"},
    {"question": "How many planets are in our solar system?", "answer": "Eight"},
    {"question": "What is the chemical symbol for mercury?", "answer": "Hg"},
    {"question": "Who wrote Don Quixote?", "answer": "Cervantes"},
    {"question": "What is the hardest natural substance?", "answer": "Diamond"},
    {"question": "What is the capital of Denmark?", "answer": "Copenhagen"},
    {"question": "How many inches are in a foot?", "answer": "12"},
    {"question": "What is Avogadro's number?", "answer": "6.022"},
    {"question": "Who wrote The Republic?", "answer": "Plato"},
    {"question": "What is 5 factorial?", "answer": "120"},
    {"question": "What is the capital of Ireland?", "answer": "Dublin"},
    {"question": "How many feet are in a mile?", "answer": "5280"},
    {"question": "What is the chemical symbol for zinc?", "answer": "Zn"},
    {"question": "Who wrote The Iliad?", "answer": "Homer"},
    {"question": "What is the largest land animal?", "answer": "Elephant"},
    {"question": "What is the capital of Austria?", "answer": "Vienna"},
    {"question": "How many centimeters are in a meter?", "answer": "100"},
    {"question": "What is the melting point of ice in Celsius?", "answer": "0"},
    {"question": "Who painted The Birth of Venus?", "answer": "Botticelli"},
    {"question": "What is 256 divided by 2?", "answer": "128"},
    {"question": "What is the capital of Switzerland?", "answer": "Bern"},
    {"question": "How many quarts are in a gallon?", "answer": "4"},
    {"question": "What is the chemical symbol for phosphorus?", "answer": "P"},
    {"question": "Who wrote The Aeneid?", "answer": "Virgil"},
    {"question": "What is the fastest land animal?", "answer": "Cheetah"},
    {"question": "What is the capital of Belgium?", "answer": "Brussels"},
    {"question": "How many ounces are in a pound?", "answer": "16"},
    {"question": "What is the atomic number of oxygen?", "answer": "8"},
    {"question": "Who wrote Candide?", "answer": "Voltaire"},
    {"question": "What is 10 to the power of 3?", "answer": "1000"},
    {"question": "What is the capital of the Netherlands?", "answer": "Amsterdam"},
    {"question": "How many cards are in a standard deck?", "answer": "52"},
    {"question": "What is the chemical symbol for sulfur?", "answer": "S"},
    {"question": "Who wrote Les Misérables?", "answer": "Victor Hugo"},
    {"question": "What is the largest moon in our solar system?", "answer": "Ganymede"},
    {"question": "What is the capital of Czech Republic?", "answer": "Prague"},
    {"question": "How many degrees in a right angle?", "answer": "90"},
    {"question": "What is the atomic number of gold?", "answer": "79"},
    {"question": "Who painted Water Lilies?", "answer": "Monet"},
    {"question": "What is 15 times 6?", "answer": "90"},
    {"question": "What is the capital of Hungary?", "answer": "Budapest"},
    {"question": "How many degrees in a full circle?", "answer": "360"},
    {"question": "What is the chemical symbol for chlorine?", "answer": "Cl"},
    {"question": "Who wrote The Stranger?", "answer": "Camus"},
    {"question": "What is the largest species of bear?", "answer": "Polar bear"},
    {"question": "What is the capital of Romania?", "answer": "Bucharest"},
    {"question": "How many minutes in a degree of latitude?", "answer": "60"},
    {"question": "What is the atomic number of uranium?", "answer": "92"},
    {"question": "Who wrote One Hundred Years of Solitude?", "answer": "Gabriel García Márquez"},
    {"question": "What is 3.14 times 2?", "answer": "6.28"},
    {"question": "What is the capital of Ukraine?", "answer": "Kyiv"},
    {"question": "How many ribs do humans have?", "answer": "24"},
    {"question": "What is the chemical symbol for magnesium?", "answer": "Mg"},
    {"question": "Who wrote The Old Man and the Sea?", "answer": "Ernest Hemingway"},
    {"question": "What is the largest big cat?", "answer": "Tiger"},
    {"question": "What is the capital of Chile?", "answer": "Santiago"},
    {"question": "How many bones in the human hand?", "answer": "27"},
    {"question": "What is the atomic number of helium?", "answer": "2"},
    {"question": "Who painted The Last Supper?", "answer": "Leonardo da Vinci"},
    {"question": "What is 1000 minus 777?", "answer": "223"},
    {"question": "What is the capital of Colombia?", "answer": "Bogotá"},
    {"question": "How many teeth do children have?", "answer": "20"},
    {"question": "What is the chemical symbol for neon?", "answer": "Ne"},
    {"question": "Who wrote Brave New World?", "answer": "Aldous Huxley"},
    {"question": "What is the largest species of penguin?", "answer": "Emperor"},
    {"question": "What is the capital of Peru?", "answer": "Lima"},
    {"question": "How many sides does a dodecahedron have?", "answer": "12"},
    {"question": "What is the atomic number of neon?", "answer": "10"},
    {"question": "Who painted Guernica?", "answer": "Pablo Picasso"},
    {"question": "What is 8 times 7?", "answer": "56"},
    {"question": "What is the capital of Malaysia?", "answer": "Kuala Lumpur"},
    {"question": "How many vertebrae in the neck?", "answer": "7"},
    {"question": "What is the chemical symbol for argon?", "answer": "Ar"},
    {"question": "Who wrote The Brothers Karamazov?", "answer": "Fyodor Dostoevsky"},
    {"question": "What is the largest fish in the world?", "answer": "Whale shark"},
    {"question": "What is the capital of the Philippines?", "answer": "Manila"},
    {"question": "How many faces does a dodecahedron have?", "answer": "12"},
    {"question": "What is the atomic number of sodium?", "answer": "11"},
    {"question": "Who wrote The Metamorphosis?", "answer": "Kafka"},
    {"question": "What is 11 times 11?", "answer": "121"},
    {"question": "What is the capital of Pakistan?", "answer": "Islamabad"},
    {"question": "How many bones in the human foot?", "answer": "26"},
    {"question": "What is the chemical symbol for helium?", "answer": "He"},
    {"question": "Who wrote The Trial?", "answer": "Kafka"},
    {"question": "What is the largest reptile?", "answer": "Saltwater crocodile"},
    {"question": "What is the capital of Bangladesh?", "answer": "Dhaka"},
    {"question": "How many elements are on the periodic table?", "answer": "118"},
    {"question": "What is the atomic number of nitrogen?", "answer": "7"},
    {"question": "Who wrote The Sound and the Fury?", "answer": "William Faulkner"},
    {"question": "What is 6 times 7?", "answer": "42"},
    {"question": "What is the capital of Nigeria?", "answer": "Abuja"},
    {"question": "How many muscles are in the human body?", "answer": "600"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "Who wrote Beloved?", "answer": "Toni Morrison"},
    {"question": "What is the largest amphibian?", "answer": "Chinese giant salamander"},
    {"question": "What is the capital of Kenya?", "answer": "Nairobi"},
    {"question": "How many valves does the heart have?", "answer": "Four"},
    {"question": "What is the atomic number of carbon?", "answer": "6"},
    {"question": "Who wrote Invisible Man?", "answer": "Ralph Ellison"},
    {"question": "What is 13 times 4?", "answer": "52"},
    {"question": "What is the capital of Morocco?", "answer": "Rabat"},
    {"question": "How many bones in the human skull?", "answer": "22"},
    {"question": "What is the chemical symbol for platinum?", "answer": "Pt"},
    {"question": "Who wrote The Color Purple?", "answer": "Alice Walker"},
    {"question": "What is the fastest bird?", "answer": "Peregrine falcon"},
    {"question": "What is the capital of Algeria?", "answer": "Algiers"},
    {"question": "How many pairs of chromosomes do humans have?", "answer": "23"},
    {"question": "What is the atomic number of iron?", "answer": "26"},
    {"question": "Who wrote Things Fall Apart?", "answer": "Chinua Achebe"},
    {"question": "What is 14 times 5?", "answer": "70"},
    {"question": "What is the capital of Saudi Arabia?", "answer": "Riyadh"},
    {"question": "How many bones in the human ribcage?", "answer": "24"},
    {"question": "What is the chemical symbol for titanium?", "answer": "Ti"},
    {"question": "Who wrote Midnight's Children?", "answer": "Salman Rushdie"},
    {"question": "What is the largest arthropod?", "answer": "Japanese spider crab"},
    {"question": "What is the capital of Iran?", "answer": "Tehran"},
    {"question": "How many bones in the human arm?", "answer": "3"},
    {"question": "What is the atomic number of copper?", "answer": "29"},
    {"question": "Who wrote The God of Small Things?", "answer": "Arundhati Roy"},
    {"question": "What is 16 plus 16?", "answer": "32"},
    {"question": "What is the capital of Iraq?", "answer": "Baghdad"},
    {"question": "How many bones in each leg?", "answer": "4"},
    {"question": "What is the chemical symbol for nickel?", "answer": "Ni"},
    {"question": "Who wrote The Kite Runner?", "answer": "Khaled Hosseini"},
    {"question": "What is the smallest mammal by weight?", "answer": "Bumblebee bat"},
    {"question": "What is the capital of Israel?", "answer": "Jerusalem"},
    {"question": "How many chambers in the heart?", "answer": "4"},
    # --- 350+ more for 600+ total ---
    {"question": "What is the capital of Scotland?", "answer": "Edinburgh"},
    {"question": "How many sides does a heptagon have?", "answer": "Seven"},
    {"question": "Who wrote Jane Eyre?", "answer": "Charlotte Brontë"},
    {"question": "What is the atomic number of helium?", "answer": "2"},
    {"question": "What is the capital of Wales?", "answer": "Cardiff"},
    {"question": "How many millimeters are in a centimeter?", "answer": "10"},
    {"question": "Who wrote Wuthering Heights?", "answer": "Emily Brontë"},
    {"question": "What is the chemical symbol for boron?", "answer": "B"},
    {"question": "What is the capital of New Zealand?", "answer": "Wellington"},
    {"question": "How many grams are in a kilogram?", "answer": "1000"},
    {"question": "Who wrote The Picture of Dorian Gray?", "answer": "Oscar Wilde"},
    {"question": "What is the atomic number of neon?", "answer": "10"},
    {"question": "What is the capital of Cuba?", "answer": "Havana"},
    {"question": "How many meters are in a kilometer?", "answer": "1000"},
    {"question": "Who wrote The Scarlet Letter?", "answer": "Nathaniel Hawthorne"},
    {"question": "What is the chemical symbol for fluorine?", "answer": "F"},
    {"question": "What is the capital of Jamaica?", "answer": "Kingston"},
    {"question": "How many pounds are in a ton?", "answer": "2000"},
    {"question": "Who wrote Uncle Tom's Cabin?", "answer": "Harriet Beecher Stowe"},
    {"question": "What is the atomic number of magnesium?", "answer": "12"},
    {"question": "What is the capital of Ghana?", "answer": "Accra"},
    {"question": "How many degrees in a straight angle?", "answer": "180"},
    {"question": "Who wrote The Grapes of Wrath?", "answer": "John Steinbeck"},
    {"question": "What is the chemical symbol for silicon?", "answer": "Si"},
    {"question": "What is the capital of Tanzania?", "answer": "Dodoma"},
    {"question": "How many zeros are in a million?", "answer": "6"},
    {"question": "Who wrote The Lord of the Rings?", "answer": "J.R.R. Tolkien"},
    {"question": "What is the atomic number of aluminum?", "answer": "13"},
    {"question": "What is the capital of Ethiopia?", "answer": "Addis Ababa"},
    {"question": "How many faces does an icosahedron have?", "answer": "20"},
    {"question": "Who wrote The Chronicles of Narnia?", "answer": "C.S. Lewis"},
    {"question": "What is the chemical symbol for manganese?", "answer": "Mn"},
    {"question": "What is the capital of Uganda?", "answer": "Kampala"},
    {"question": "How many sides does a nonagon have?", "answer": "Nine"},
    {"question": "Who wrote Alice in Wonderland?", "answer": "Lewis Carroll"},
    {"question": "What is the atomic number of phosphorus?", "answer": "15"},
    {"question": "What is the capital of Senegal?", "answer": "Dakar"},
    {"question": "How many legs does a lobster have?", "answer": "10"},
    {"question": "Who wrote Robinson Crusoe?", "answer": "Daniel Defoe"},
    {"question": "What is the chemical symbol for cobalt?", "answer": "Co"},
    {"question": "What is the capital of Zimbabwe?", "answer": "Harare"},
    {"question": "How many sides does a decagon have?", "answer": "10"},
    {"question": "Who wrote Gulliver's Travels?", "answer": "Jonathan Swift"},
    {"question": "What is the atomic number of sulfur?", "answer": "16"},
    {"question": "What is the capital of Nepal?", "answer": "Kathmandu"},
    {"question": "How many eyes does a bee have?", "answer": "5"},
    {"question": "Who wrote The Three Musketeers?", "answer": "Alexandre Dumas"},
    {"question": "What is the chemical symbol for chromium?", "answer": "Cr"},
    {"question": "What is the capital of Sri Lanka?", "answer": "Colombo"},
    {"question": "How many bones in the human finger?", "answer": "3"},
    {"question": "Who wrote The Count of Monte Cristo?", "answer": "Alexandre Dumas"},
    {"question": "What is the atomic number of chlorine?", "answer": "17"},
    {"question": "What is the capital of Cambodia?", "answer": "Phnom Penh"},
    {"question": "How many wings does a bee have?", "answer": "4"},
    {"question": "Who wrote Madame Bovary?", "answer": "Gustave Flaubert"},
    {"question": "What is the chemical symbol for vanadium?", "answer": "V"},
    {"question": "What is the capital of Myanmar?", "answer": "Naypyidaw"},
    {"question": "How many teeth does a great white shark have?", "answer": "300"},
    {"question": "Who wrote The Hunchback of Notre-Dame?", "answer": "Victor Hugo"},
    {"question": "What is the atomic number of argon?", "answer": "18"},
    {"question": "What is the capital of Laos?", "answer": "Vientiane"},
    {"question": "How many hearts does an octopus have?", "answer": "3"},
    {"question": "Who wrote The Phantom of the Opera?", "answer": "Gaston Leroux"},
    {"question": "What is the chemical symbol for scandium?", "answer": "Sc"},
    {"question": "What is the capital of Mongolia?", "answer": "Ulaanbaatar"},
    {"question": "How many stomachs does a cow have?", "answer": "4"},
    {"question": "Who wrote The Jungle Book?", "answer": "Rudyard Kipling"},
    {"question": "What is the atomic number of potassium?", "answer": "19"},
    {"question": "What is the capital of Kazakhstan?", "answer": "Astana"},
    {"question": "How many arms does a starfish have?", "answer": "5"},
    {"question": "Who wrote Treasure Island?", "answer": "Robert Louis Stevenson"},
    {"question": "What is the chemical symbol for titanium?", "answer": "Ti"},
    {"question": "What is the capital of Uzbekistan?", "answer": "Tashkent"},
    {"question": "How many bones in the human ear?", "answer": "3"},
    {"question": "Who wrote Dr. Jekyll and Mr. Hyde?", "answer": "Robert Louis Stevenson"},
    {"question": "What is the atomic number of calcium?", "answer": "20"},
    {"question": "What is the capital of Azerbaijan?", "answer": "Baku"},
    {"question": "How many teeth does an adult dog have?", "answer": "42"},
    {"question": "Who wrote The Time Machine?", "answer": "H.G. Wells"},
    {"question": "What is the chemical symbol for iron?", "answer": "Fe"},
    {"question": "What is the capital of Georgia?", "answer": "Tbilisi"},
    {"question": "How many bones does a giraffe have in its neck?", "answer": "7"},
    {"question": "Who wrote The War of the Worlds?", "answer": "H.G. Wells"},
    {"question": "What is the atomic number of scandium?", "answer": "21"},
    {"question": "What is the capital of Armenia?", "answer": "Yerevan"},
    {"question": "How many teeth does an adult cat have?", "answer": "30"},
    {"question": "Who wrote The Invisible Man?", "answer": "H.G. Wells"},
    {"question": "What is the chemical symbol for nickel?", "answer": "Ni"},
    {"question": "What is the capital of Lebanon?", "answer": "Beirut"},
    {"question": "How many bones in a human baby?", "answer": "300"},
    {"question": "Who wrote Around the World in Eighty Days?", "answer": "Jules Verne"},
    {"question": "What is the atomic number of titanium?", "answer": "22"},
    {"question": "What is the capital of Syria?", "answer": "Damascus"},
    {"question": "How many teeth does a snail have?", "answer": "14000"},
    {"question": "Who wrote Twenty Thousand Leagues Under the Sea?", "answer": "Jules Verne"},
    {"question": "What is the chemical symbol for zinc?", "answer": "Zn"},
    {"question": "What is the capital of Jordan?", "answer": "Amman"},
    {"question": "How many bones in the human face?", "answer": "14"},
    {"question": "Who wrote Journey to the Center of the Earth?", "answer": "Jules Verne"},
    {"question": "What is the atomic number of vanadium?", "answer": "23"},
    {"question": "What is the capital of Yemen?", "answer": "Sana'a"},
    {"question": "How many ribs does a human have?", "answer": "24"},
    {"question": "Who wrote The Adventures of Tom Sawyer?", "answer": "Mark Twain"},
    {"question": "What is the chemical symbol for gallium?", "answer": "Ga"},
    {"question": "What is the capital of Oman?", "answer": "Muscat"},
    {"question": "How many vertebrae in the lumbar spine?", "answer": "5"},
    {"question": "Who wrote The Adventures of Huckleberry Finn?", "answer": "Mark Twain"},
    {"question": "What is the atomic number of chromium?", "answer": "24"},
    {"question": "What is the capital of Qatar?", "answer": "Doha"},
    {"question": "How many bones in the thoracic spine?", "answer": "12"},
    {"question": "Who wrote Little Women?", "answer": "Louisa May Alcott"},
    {"question": "What is the chemical symbol for germanium?", "answer": "Ge"},
    {"question": "What is the capital of Kuwait?", "answer": "Kuwait City"},
    {"question": "How many cervical vertebrae are there?", "answer": "7"},
    {"question": "Who wrote Little Men?", "answer": "Louisa May Alcott"},
    {"question": "What is the atomic number of manganese?", "answer": "25"},
    {"question": "What is the capital of Bahrain?", "answer": "Manama"},
    {"question": "How many bones in the human pelvis?", "answer": "3"},
    {"question": "Who wrote The Wizard of Oz?", "answer": "L. Frank Baum"},
    {"question": "What is the chemical symbol for arsenic?", "answer": "As"},
    {"question": "What is the capital of Cyprus?", "answer": "Nicosia"},
    {"question": "How many bones in the human ankle?", "answer": "7"},
    {"question": "Who wrote Peter Pan?", "answer": "J.M. Barrie"},
    {"question": "What is the atomic number of iron?", "answer": "26"},
    {"question": "What is the capital of Iceland?", "answer": "Reykjavik"},
    {"question": "How many bones in the human wrist?", "answer": "8"},
    {"question": "Who wrote Winnie-the-Pooh?", "answer": "A.A. Milne"},
    {"question": "What is the chemical symbol for selenium?", "answer": "Se"},
    {"question": "What is the capital of Luxembourg?", "answer": "Luxembourg City"},
    {"question": "How many bones in the human knee?", "answer": "4"},
    {"question": "Who wrote The Wind in the Willows?", "answer": "Kenneth Grahame"},
    {"question": "What is the atomic number of cobalt?", "answer": "27"},
    {"question": "What is the capital of Malta?", "answer": "Valletta"},
    {"question": "How many bones in the human elbow?", "answer": "3"},
    {"question": "Who wrote Charlotte's Web?", "answer": "E.B. White"},
    {"question": "What is the chemical symbol for bromine?", "answer": "Br"},
    {"question": "What is the capital of Estonia?", "answer": "Tallinn"},
    {"question": "How many bones in the human shoulder?", "answer": "4"},
    {"question": "Who wrote Stuart Little?", "answer": "E.B. White"},
    {"question": "What is the atomic number of nickel?", "answer": "28"},
    {"question": "What is the capital of Latvia?", "answer": "Riga"},
    {"question": "How many bones in the human hip?", "answer": "2"},
    {"question": "Who wrote The Secret Garden?", "answer": "Frances Hodgson Burnett"},
    {"question": "What is the chemical symbol for krypton?", "answer": "Kr"},
    {"question": "What is the capital of Lithuania?", "answer": "Vilnius"},
    {"question": "How many bones in the human jaw?", "answer": "2"},
    {"question": "Who wrote A Little Princess?", "answer": "Frances Hodgson Burnett"},
    {"question": "What is the atomic number of copper?", "answer": "29"},
    {"question": "What is the capital of Slovenia?", "answer": "Ljubljana"},
    {"question": "How many bones in the human nose?", "answer": "2"},
    {"question": "Who wrote Black Beauty?", "answer": "Anna Sewell"},
    {"question": "What is the chemical symbol for rubidium?", "answer": "Rb"},
    {"question": "What is the capital of Croatia?", "answer": "Zagreb"},
    {"question": "How many bones in the human collarbone?", "answer": "2"},
    {"question": "Who wrote Heidi?", "answer": "Johanna Spyri"},
    {"question": "What is the atomic number of zinc?", "answer": "30"},
    {"question": "What is the capital of Serbia?", "answer": "Belgrade"},
    {"question": "How many bones in the human shoulder blade?", "answer": "2"},
    {"question": "Who wrote Anne of Green Gables?", "answer": "L.M. Montgomery"},
    {"question": "What is the chemical symbol for strontium?", "answer": "Sr"},
    {"question": "What is the capital of Bulgaria?", "answer": "Sofia"},
    {"question": "What is 18 times 5?", "answer": "90"},
    {"question": "Who wrote The Call of the Wild?", "answer": "Jack London"},
    {"question": "What is the atomic number of gallium?", "answer": "31"},
    {"question": "What is the capital of Slovakia?", "answer": "Bratislava"},
    {"question": "What is 19 times 4?", "answer": "76"},
    {"question": "Who wrote White Fang?", "answer": "Jack London"},
    {"question": "What is the chemical symbol for yttrium?", "answer": "Y"},
    {"question": "What is the capital of Bosnia?", "answer": "Sarajevo"},
    {"question": "What is 21 times 3?", "answer": "63"},
    {"question": "Who wrote The Raven?", "answer": "Edgar Allan Poe"},
    {"question": "What is the atomic number of germanium?", "answer": "32"},
    {"question": "What is the capital of Albania?", "answer": "Tirana"},
    {"question": "What is 22 times 2?", "answer": "44"},
    {"question": "Who wrote The Tell-Tale Heart?", "answer": "Edgar Allan Poe"},
    {"question": "What is the chemical symbol for zirconium?", "answer": "Zr"},
    {"question": "What is the capital of North Macedonia?", "answer": "Skopje"},
    {"question": "What is 23 times 4?", "answer": "92"},
    {"question": "Who wrote Moby-Dick?", "answer": "Herman Melville"},
    {"question": "What is the atomic number of arsenic?", "answer": "33"},
    {"question": "What is the capital of Montenegro?", "answer": "Podgorica"},
    {"question": "What is 24 times 5?", "answer": "120"},
    {"question": "Who wrote Bartleby the Scrivener?", "answer": "Herman Melville"},
    {"question": "What is the chemical symbol for niobium?", "answer": "Nb"},
    {"question": "What is the capital of Kosovo?", "answer": "Pristina"},
    {"question": "What is 25 times 5?", "answer": "125"},
    {"question": "Who wrote The Fall of the House of Usher?", "answer": "Edgar Allan Poe"},
    {"question": "What is the atomic number of selenium?", "answer": "34"},
    {"question": "What is the capital of Moldova?", "answer": "Chișinău"},
    {"question": "What is 30 divided by 5?", "answer": "6"},
    {"question": "Who wrote The Cask of Amontillado?", "answer": "Edgar Allan Poe"},
    {"question": "What is the chemical symbol for molybdenum?", "answer": "Mo"},
    {"question": "What is the capital of Belarus?", "answer": "Minsk"},
    {"question": "What is 36 divided by 6?", "answer": "6"},
    {"question": "Who wrote Leaves of Grass?", "answer": "Walt Whitman"},
    {"question": "What is the atomic number of bromine?", "answer": "35"},
    {"question": "What is the capital of Lithuania?", "answer": "Vilnius"},
    {"question": "What is 49 divided by 7?", "answer": "7"},
    {"question": "Who wrote Song of Myself?", "answer": "Walt Whitman"},
    {"question": "What is the chemical symbol for technetium?", "answer": "Tc"},
    {"question": "What is the capital of Ecuador?", "answer": "Quito"},
    {"question": "What is 64 divided by 8?", "answer": "8"},
    {"question": "Who wrote The Road Not Taken?", "answer": "Robert Frost"},
    {"question": "What is the atomic number of krypton?", "answer": "36"},
    {"question": "What is the capital of Bolivia?", "answer": "La Paz"},
    {"question": "What is 81 divided by 9?", "answer": "9"},
    {"question": "Who wrote Stopping by Woods on a Snowy Evening?", "answer": "Robert Frost"},
    {"question": "What is the chemical symbol for ruthenium?", "answer": "Ru"},
    {"question": "What is the capital of Paraguay?", "answer": "Asunción"},
    {"question": "What is 100 divided by 5?", "answer": "20"},
    {"question": "Who wrote The Waste Land?", "answer": "T.S. Eliot"},
    {"question": "What is the atomic number of rubidium?", "answer": "37"},
    {"question": "What is the capital of Uruguay?", "answer": "Montevideo"},
    {"question": "What is 121 divided by 11?", "answer": "11"},
    {"question": "Who wrote The Love Song of J. Alfred Prufrock?", "answer": "T.S. Eliot"},
    {"question": "What is the chemical symbol for rhodium?", "answer": "Rh"},
    {"question": "What is the capital of Venezuela?", "answer": "Caracas"},
    {"question": "What is 169 square root?", "answer": "13"},
    {"question": "Who wrote Howl?", "answer": "Allen Ginsberg"},
    {"question": "What is the atomic number of strontium?", "answer": "38"},
    {"question": "What is the capital of Costa Rica?", "answer": "San José"},
    {"question": "What is 196 square root?", "answer": "14"},
    {"question": "Who wrote The Bell Jar?", "answer": "Sylvia Plath"},
    {"question": "What is the chemical symbol for palladium?", "answer": "Pd"},
    {"question": "What is the capital of Panama?", "answer": "Panama City"},
    {"question": "What is 225 square root?", "answer": "15"},
    {"question": "Who wrote Ariel?", "answer": "Sylvia Plath"},
    {"question": "What is the atomic number of yttrium?", "answer": "39"},
    {"question": "What is the capital of Guatemala?", "answer": "Guatemala City"},
    {"question": "What is 256 square root?", "answer": "16"},
    {"question": "Who wrote Slaughterhouse-Five?", "answer": "Kurt Vonnegut"},
    {"question": "What is the chemical symbol for silver?", "answer": "Ag"},
    {"question": "What is the capital of Honduras?", "answer": "Tegucigalpa"},
    {"question": "What is 289 square root?", "answer": "17"},
    {"question": "Who wrote Cat's Cradle?", "answer": "Kurt Vonnegut"},
    {"question": "What is the atomic number of zirconium?", "answer": "40"},
    {"question": "What is the capital of Nicaragua?", "answer": "Managua"},
    {"question": "What is 324 square root?", "answer": "18"},
    {"question": "Who wrote Breakfast of Champions?", "answer": "Kurt Vonnegut"},
    {"question": "What is the chemical symbol for cadmium?", "answer": "Cd"},
    {"question": "What is the capital of El Salvador?", "answer": "San Salvador"},
    {"question": "What is 361 square root?", "answer": "19"},
    {"question": "Who wrote One Flew Over the Cuckoo's Nest?", "answer": "Ken Kesey"},
    {"question": "What is the atomic number of niobium?", "answer": "41"},
    {"question": "What is the capital of Dominican Republic?", "answer": "Santo Domingo"},
    {"question": "What is 400 square root?", "answer": "20"},
    {"question": "Who wrote Sometimes a Great Notion?", "answer": "Ken Kesey"},
    {"question": "What is the chemical symbol for indium?", "answer": "In"},
    {"question": "What is the capital of Haiti?", "answer": "Port-au-Prince"},
    {"question": "What is 2 cubed?", "answer": "8"},
    {"question": "Who wrote Catch-22?", "answer": "Joseph Heller"},
    {"question": "What is the atomic number of molybdenum?", "answer": "42"},
    {"question": "What is the capital of Trinidad and Tobago?", "answer": "Port of Spain"},
    {"question": "What is 3 cubed?", "answer": "27"},
    {"question": "Who wrote Something Happened?", "answer": "Joseph Heller"},
    {"question": "What is the chemical symbol for tin?", "answer": "Sn"},
    {"question": "What is the capital of Barbados?", "answer": "Bridgetown"},
    {"question": "What is 4 cubed?", "answer": "64"},
    {"question": "Who wrote The Naked and the Dead?", "answer": "Norman Mailer"},
    {"question": "What is the atomic number of technetium?", "answer": "43"},
    {"question": "What is the capital of Bahamas?", "answer": "Nassau"},
    {"question": "What is 5 cubed?", "answer": "125"},
    {"question": "Who wrote The Executioner's Song?", "answer": "Norman Mailer"},
    {"question": "What is the chemical symbol for antimony?", "answer": "Sb"},
    {"question": "What is the capital of Botswana?", "answer": "Gaborone"},
    {"question": "What is 6 cubed?", "answer": "216"},
    {"question": "Who wrote In Cold Blood?", "answer": "Truman Capote"},
    {"question": "What is the atomic number of ruthenium?", "answer": "44"},
    {"question": "What is the capital of Namibia?", "answer": "Windhoek"},
    {"question": "What is 7 cubed?", "answer": "343"},
    {"question": "Who wrote Breakfast at Tiffany's?", "answer": "Truman Capote"},
    {"question": "What is the chemical symbol for tellurium?", "answer": "Te"},
    {"question": "What is the capital of Zambia?", "answer": "Lusaka"},
    {"question": "What is 8 cubed?", "answer": "512"},
    {"question": "Who wrote The Heart Is a Lonely Hunter?", "answer": "Carson McCullers"},
    {"question": "What is the atomic number of rhodium?", "answer": "45"},
    {"question": "What is the capital of Malawi?", "answer": "Lilongwe"},
    {"question": "What is 9 cubed?", "answer": "729"},
    {"question": "Who wrote The Member of the Wedding?", "answer": "Carson McCullers"},
    {"question": "What is the chemical symbol for iodine?", "answer": "I"},
    {"question": "What is the capital of Mozambique?", "answer": "Maputo"},
    {"question": "What is 10 cubed?", "answer": "1000"},
    {"question": "Who wrote The Glass Menagerie?", "answer": "Tennessee Williams"},
    {"question": "What is the atomic number of palladium?", "answer": "46"},
    {"question": "What is the capital of Angola?", "answer": "Luanda"},
    {"question": "In what year did World War I begin?", "answer": "1914"},
    {"question": "Who wrote A Streetcar Named Desire?", "answer": "Tennessee Williams"},
    {"question": "What is the chemical symbol for xenon?", "answer": "Xe"},
    {"question": "What is the capital of Cameroon?", "answer": "Yaoundé"},
    {"question": "In what year did the American Civil War end?", "answer": "1865"},
    {"question": "Who wrote Death of a Salesman?", "answer": "Arthur Miller"},
    {"question": "What is the atomic number of silver?", "answer": "47"},
    {"question": "What is the capital of Ivory Coast?", "answer": "Yamoussoukro"},
    {"question": "In what year did man first land on the Moon?", "answer": "1969"},
    {"question": "Who wrote The Crucible?", "answer": "Arthur Miller"},
    {"question": "What is the chemical symbol for cesium?", "answer": "Cs"},
    {"question": "What is the capital of Tunisia?", "answer": "Tunis"},
    {"question": "In what year did the Berlin Wall fall?", "answer": "1989"},
    {"question": "Who wrote Long Day's Journey Into Night?", "answer": "Eugene O'Neill"},
    {"question": "What is the atomic number of cadmium?", "answer": "48"},
    {"question": "What is the capital of Libya?", "answer": "Tripoli"},
    {"question": "In what year did Columbus reach the Americas?", "answer": "1492"},
    {"question": "Who wrote The Iceman Cometh?", "answer": "Eugene O'Neill"},
    {"question": "What is the chemical symbol for barium?", "answer": "Ba"},
    {"question": "What is the capital of Sudan?", "answer": "Khartoum"},
    {"question": "In what year did the French Revolution begin?", "answer": "1789"},
    {"question": "Who wrote Waiting for Godot?", "answer": "Samuel Beckett"},
    {"question": "What is the atomic number of indium?", "answer": "49"},
    {"question": "What is the capital of Somalia?", "answer": "Mogadishu"},
    {"question": "In what year did India gain independence?", "answer": "1947"},
    {"question": "Who wrote Endgame?", "answer": "Samuel Beckett"},
    {"question": "What is the chemical symbol for lanthanum?", "answer": "La"},
    {"question": "What is the capital of Madagascar?", "answer": "Antananarivo"},
    {"question": "In what year did apartheid end in South Africa?", "answer": "1994"},
    {"question": "Who wrote Rosencrantz and Guildenstern Are Dead?", "answer": "Tom Stoppard"},
    {"question": "What is the atomic number of tin?", "answer": "50"},
    {"question": "What is the capital of Rwanda?", "answer": "Kigali"},
    {"question": "How many US Supreme Court justices are there?", "answer": "9"},
    {"question": "Who wrote The Importance of Being Earnest?", "answer": "Oscar Wilde"},
    {"question": "What is the chemical symbol for cerium?", "answer": "Ce"},
    {"question": "What is the capital of Mauritius?", "answer": "Port Louis"},
    {"question": "How many amendments are in the US Bill of Rights?", "answer": "10"},
    {"question": "Who wrote Pygmalion?", "answer": "George Bernard Shaw"},
    {"question": "What is the atomic number of antimony?", "answer": "51"},
    {"question": "What is the capital of Seychelles?", "answer": "Victoria"},
    {"question": "How many strings does a violin have?", "answer": "4"},
    {"question": "Who wrote Saint Joan?", "answer": "George Bernard Shaw"},
    {"question": "What is the chemical symbol for praseodymium?", "answer": "Pr"},
    {"question": "What is the capital of Fiji?", "answer": "Suva"},
    {"question": "How many strings does a guitar have?", "answer": "6"},
    {"question": "Who wrote Arms and the Man?", "answer": "George Bernard Shaw"},
    {"question": "What is the atomic number of tellurium?", "answer": "52"},
    {"question": "What is the capital of Papua New Guinea?", "answer": "Port Moresby"},
    {"question": "How many strings does a cello have?", "answer": "4"},
    {"question": "Who wrote Man and Superman?", "answer": "George Bernard Shaw"},
    {"question": "What is the chemical symbol for neodymium?", "answer": "Nd"},
    {"question": "What is the capital of Solomon Islands?", "answer": "Honiara"},
    {"question": "How many strings does a harp have?", "answer": "47"},
    {"question": "Who wrote Mrs. Warren's Profession?", "answer": "George Bernard Shaw"},
    {"question": "What is the atomic number of iodine?", "answer": "53"},
    {"question": "What is the capital of Tonga?", "answer": "Nuku'alofa"},
    {"question": "How many keys on a standard computer keyboard?", "answer": "104"},
    {"question": "Who wrote Candida?", "answer": "George Bernard Shaw"},
    {"question": "What is the chemical symbol for promethium?", "answer": "Pm"},
    {"question": "What is the capital of Samoa?", "answer": "Apia"},
]


def load_unambiguous_sample(n: int) -> list:
    """Load n unambiguous-style questions: try Natural Questions first, else fallback to 200 factual Q&As."""
    from datasets import load_dataset

    skip_prefixes = (
        "is ", "are ", "was ", "were ", "did ", "do ", "does ",
        "can ", "has ", "have ",
    )

    result = []
    try:
        ds = load_dataset("natural_questions", "default", split="train")
        ds = ds.shuffle(seed=42)
        for item in ds:
            if len(result) >= n:
                break
            ann = item.get("annotations")
            if ann is None:
                continue
            # annotations can be dict or list (one per question)
            if isinstance(ann, list):
                ann = ann[0] if ann else {}
            short_answers = ann.get("short_answers") if isinstance(ann, dict) else []
            if not short_answers or len(short_answers) == 0:
                continue
            first = short_answers[0]
            texts = first.get("text", []) if isinstance(first, dict) else []
            if not texts:
                continue
            answer_text = (texts[0] if isinstance(texts[0], str) else str(texts[0])).strip()
            word_count = len(answer_text.split())
            if word_count < 1 or word_count > 5:
                continue
            q = item.get("question")
            if q is None:
                continue
            if isinstance(q, list):
                q = q[0] if q else {}
            qtext = q.get("text", "") if isinstance(q, dict) else ""
            if not qtext:
                continue
            q_lower = qtext.strip().lower()
            if any(q_lower.startswith(p) for p in skip_prefixes):
                continue
            result.append({"question": qtext.strip(), "answer": answer_text})
        if len(result) >= n:
            print("Using Natural Questions for unambiguous set")
            return result[:n]
    except Exception:
        pass

    # Fallback: hardcoded factual Q&As
    random.seed(42)
    fallback = _FALLBACK_UNAMBIGUOUS_QA.copy()
    random.shuffle(fallback)
    result = fallback[:n]
    print("Using fallback factual questions for unambiguous set")
    return result


def load_pop_qa_sample(n: int) -> list:
    """Return n shuffled PopQA questions."""
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test")
    ds = ds.shuffle(seed=42)

    result = []
    for item in ds:
        if len(result) >= n:
            break
        result.append({"question": item["question"], "answer": item["obj"]})
    return result


def measure_answer_entropy(
    model,
    tokenizer,
    question: str,
    n_samples: int,
    temperature: float,
    max_new_tokens: int = 5,
) -> float:
    """
    Estimate the model's answer entropy for a question by sampling n_samples
    completions and computing normalized Shannon entropy over first-word answers.

    Returns a float in [0, 1]: 0 = deterministic, 1 = maximally random.
    """
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("mps")
    attention_mask = inputs["attention_mask"].to("mps")

    answers = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(generated.cpu(), skip_special_tokens=True)
        first_word = decoded.strip().split()[0].lower() if decoded.strip() else ""
        answers.append(first_word)
        torch.mps.empty_cache()

    counts = Counter(answers)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])

    # Normalized Shannon entropy: 0 = all same, ~1 = all different
    if len(probs) == 1:
        return 0.0
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(n_samples)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def build_ambiguous_unambiguous_split(
    model,
    tokenizer,
    unambiguous_questions: list,
    pop_questions: list,
    n_per_condition: int,
    entropy_high: float,
    entropy_low: float,
    results_dir: str,
):
    """
    Partition questions into ambiguous (high entropy) and unambiguous (low entropy)
    sets by sampling model completions. Results are cached to disk.
    """
    cache_path = os.path.join(results_dir, "dataset_splits.json")
    if os.path.exists(cache_path):
        print(f"Loading cached dataset split from {cache_path}")
        with open(cache_path) as f:
            splits = json.load(f)
        return splits["ambiguous"], splits["unambiguous"]

    os.makedirs(results_dir, exist_ok=True)

    popqa_cache_path = os.path.join(results_dir, "popqa_entropy_cache.json")
    popqa_ckpt_path  = os.path.join(results_dir, "popqa_entropy_cache.json.ckpt")

    if os.path.exists(popqa_cache_path):
        print(f"Loading PopQA entropy from cache: {popqa_cache_path}")
        with open(popqa_cache_path) as f:
            pop_with_entropy = json.load(f)
    else:
        # Resume from per-10-item checkpoint if available
        if os.path.exists(popqa_ckpt_path):
            with open(popqa_ckpt_path) as f:
                pop_with_entropy = json.load(f)
            already_measured = {x["question"] for x in pop_with_entropy}
            print(
                f"Resuming PopQA entropy measurement from checkpoint "
                f"({len(pop_with_entropy)} items done)"
            )
        else:
            pop_with_entropy = []
            already_measured = set()

        remaining = [q for q in pop_questions if q["question"] not in already_measured]
        print(
            f"Measuring entropy for PopQA questions "
            f"({len(remaining)} remaining of {len(pop_questions)})..."
        )
        for i, item in enumerate(tqdm(remaining)):
            e = measure_answer_entropy(
                model,
                tokenizer,
                item["question"],
                N_SAMPLES_FOR_ENTROPY,
                TEMPERATURE_SAMPLING,
            )
            pop_with_entropy.append({**item, "entropy": e})

            if (i + 1) % 10 == 0:
                with open(popqa_ckpt_path, "w") as f:
                    json.dump(pop_with_entropy, f, indent=2)

        # Promote to final cache and remove checkpoint
        with open(popqa_cache_path, "w") as f:
            json.dump(pop_with_entropy, f, indent=2)
        print(f"Saved PopQA entropy cache to {popqa_cache_path}")
        if os.path.exists(popqa_ckpt_path):
            os.remove(popqa_ckpt_path)

    # Both conditions are drawn from PopQA, split by entropy threshold.
    # The `unambiguous_questions` parameter is intentionally ignored so that
    # domain and question format are held constant across conditions.
    ambiguous   = [x for x in pop_with_entropy if x["entropy"] >  entropy_high]
    unambiguous = [x for x in pop_with_entropy if x["entropy"] <  entropy_low]

    # Print entropy distribution to help tune thresholds
    all_entropies = np.array([x["entropy"] for x in pop_with_entropy])
    print(
        f"\nPopQA entropy distribution (n={len(all_entropies)}):\n"
        f"  min={all_entropies.min():.3f}  "
        f"p25={np.percentile(all_entropies, 25):.3f}  "
        f"median={np.median(all_entropies):.3f}  "
        f"p75={np.percentile(all_entropies, 75):.3f}  "
        f"max={all_entropies.max():.3f}\n"
        f"  entropy_high threshold={entropy_high}  "
        f"entropy_low threshold={entropy_low}"
    )
    print(f"Found {len(ambiguous)} ambiguous, {len(unambiguous)} unambiguous questions")

    assert len(ambiguous) >= n_per_condition, (
        f"Not enough ambiguous questions. Found {len(ambiguous)}, need {n_per_condition}. "
        "Lower ENTROPY_THRESHOLD_HIGH."
    )
    if len(unambiguous) < n_per_condition:
        print(f"WARNING: Only {len(unambiguous)} unambiguous questions found.")
        print(f"Reducing n_per_condition to {len(unambiguous)}")
        n_per_condition = len(unambiguous)

    random.seed(42)
    ambiguous = random.sample(ambiguous, n_per_condition)
    unambiguous = random.sample(unambiguous, n_per_condition)

    with open(cache_path, "w") as f:
        json.dump({"ambiguous": ambiguous, "unambiguous": unambiguous}, f, indent=2)
    print(f"Saved dataset split to {cache_path}")

    return ambiguous, unambiguous
