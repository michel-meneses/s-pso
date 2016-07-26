// Michel Conrado Cardoso Meneses (30/10/2014 - 16:57h)

#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>
#include <windows.h>

// DEFINIÇÃO DE CONSTANTES

#define quant_val 80																												// quantidade máxima de valores discretos assumidos por um atributo;
#define tam_val 60																													// tamanho máximo do nome de um valor;
#define quant_max_atrib 100																											// quantidade máxima de atributos do arquivo;
#define quant_mat_cont 8																											// quantidade de posições no vetor da matriz de contigência;
#define quant_func_ob 8																												// quantidade de posições no vetor das funções objetivo;
#define quant_mat_conf 4																											// quantidade de posições no vetor da matriz de confusão;

/* posições da matriz de contingência */
#define BH 0																														// B verdade e H verdade;
#define _BH 1																														// B falso e H verdade;
#define B_H 2																														// B verdade e H falso;
#define _B_H 3																														// B falso e H falso;
#define B 4																															// B verdade;
#define _B 5																														// B falso;
#define H 6																															// H verdade;
#define _H 7																														// H falso;

/* índices das funções objetivo */
#define ACC 0																														// precisão;
#define ERR 1																														// erro;
#define NEGREL 2																													// confiança negativa;
#define ACCLP 3																														// precisão de Laplace;
#define SENS 4																														// sensitividade;
#define SPEC 5																														// cobertura;
#define COV 6																														// suporte;
#define SUP 7

/* posições da matriz de confusão */
#define TP 0																														// positivo verdadeiro
#define FP 1																														// falso positivo
#define TN 2																														// negativo verdadeiro
#define FN 3																														// falso negativo

__device__ const double MAX_PHI = 1;
__device__ const double MAX_OMEGA = 0.8;
__device__ const int INFINITO = 1000000;
const int MENOS_INFINITO = -1000000;

// DEFINIÇÃO DE ESTRUTURAS

struct atributo{

	char* nome;																													// nome do atributo
	int cod[quant_val];																											// código numérico dos valores do atributo;
	char valor[quant_val][tam_val];																								// valores do atributo;
	int quant_real;																												// quantidade real de valores que um atributo assume;
	int numerico;																												// se atributo for numérico a flag será 1;
};
typedef struct atributo atributo;

struct exemplo{

	int campos[quant_val];																										// vetor de atributos do exemplo;
	int quant_real;																												// quantidade real de atributos do exemplo;
};
typedef struct exemplo exemplo;

struct regra{

	int valores[quant_max_atrib];
	int mat_cont[quant_mat_cont];																								// vetor da matriz de contingência;
	double func_ob[quant_func_ob];																								// vetor contendo o valor das funções objetivo da regra;
	int nula;																													// booleano que indica se a regra é nula (1) ou não (0);
	double crowding_distance;
	int quant_dominadores;																										// quantidade de regras que dominam por Pareto esta regra;
};
typedef struct regra regra;

struct classificador{

	regra* regras;																												// regras que compõem o classificador;
	int quant_regras;																											// quantidade de regras que compõem o classificador;
	int mat_conf[quant_mat_conf];																								// matriz de confusão do classificador;
};
typedef struct classificador classificador;

struct parametros{

	char* arquivo;																												// diretório do arquivo de treino .arff;
	int execucoes;																												// número de execuções do algoritmo;
	int classe;																													// define se regras geradas serão para classe positiva (1), negativa (0) ou ambas (-1);	
	int funcao_obj;																												// função objetivo a ser usada na B.L;	
	int bl_interacoes;																											// nº de interações do algoritmo de busca local;
	int bl_vizinhos;																											// nº de vizinhos a serem gerados em cada interação do algoritmo de busca local;
	char* funcoes_obj_pareto;																									// sequência de bits que indicam as funções objetivos a serem consideradas na dominância de Pareto;
	int quant_regras_pareto;																									// nº de regras que serão geradas e analisadas quanto à dominância de pareto; 
	int quant_particulas;																										// quantidade de partículas de um enxame;
	int quant_enxames;																											// quantidade de enxames criados
	int tamanho_arquivo;																										// quantidade máxima de soluções do S-PSO
	int metodo_dopagem_solucao;																									// forma de inserir regras dominadas temporariamente no arquivo solução;
	int metodo_gerar_regras;																									// forma de gerar regras aleatórias. Gerar regras uniformimente aleatórias = 0, privilegiar o valor "_" = 1 (caso seja 1, atribuir um valor ao campo "prob_valor_vazio");
	int prob_valor_vazio;																										// probabilidade do valor vazio ser atribuído a um atributo de uma regra gerada aleatoriamente (só é usado caso o valor do campo "metodo_gerar_regras" seja 1);
};
typedef struct parametros parametros;

struct regiao_pareto{

	regra* solucoes;																											// conjunto de regras não dominadas;
	int func_obj[quant_func_ob];																								// funções objetivos consideradas para verificação de dominância;
	int quant_sol_pareto;																										// quantidade de regras contidas no conjunto não dominado de pareto;
};
typedef struct regiao_pareto regiao_pareto;

struct particula{

	regra posicao;
	regra lBest;
	regra gBest;
	double** velocidade;
};
typedef struct particula particula;

// DECLARAÇÃO DE FUNÇÕES

/* Função para carrear arquivo .arff */
FILE* carregarArq(char argv[]){

	FILE* input = fopen(argv, "r");

	if (input == NULL){																											// caso o  arquivo não exista

		printf("Arquivo não encontrado!\n");																					// emite mensagem de erro
		printf("\n%s", argv);
		printf("\nlenght = %d", strlen(argv));
		system("pause");
		exit(1);																												// e finaliza
	}
	return input;
}

/* Função para carrear os arquivos .arff de exemplos particionados.
Considera-se que existam 10 partições, sendo o parâmetro número referente
à partição que não será utilizada para o treinamento (apenas para o teste). */
FILE* carregarArqDados(char nomeBase[], int numero){

	//Cria arquivo que receberá o conjunto dos exemplos de todas as partições consideradas
	//para o treinamento. Além disso, copia o cabeçalho do arquivo da 1ª partição, de modo
	//que o arquivo final possa ter seus atributos processados. 
	FILE* dados = fopen("conjunto_particoes_dados.txt", "w");
	if (dados == NULL){

		printf("Nao foi possivel criar arquivo!\n");
		printf("\n%s", "conjunto_particoes_dados.txt");
		system("pause");
		exit(1);
	}

	//Constroi nome completo do diretório da partição 0.
	char nomeArquivo[40];
	sprintf(nomeArquivo, "bases/%s/it%d/%s_data.arff", nomeBase, 0, nomeBase);
	FILE* input = fopen(nomeArquivo, "r");

	//Informa possível falha na abertura da partição. 
	if (input == NULL){
		printf("Arquivo não encontrado!\n");
		printf("\n%s", nomeArquivo);
		printf("\nlenght = %d", strlen(nomeArquivo));
		system("pause");
		exit(1);
	}

	//Copia cada linha do arquivo da partição 0, até que todo o cabeçalho tenha sido copiado. 
	const int tamanhoMaximoLinha = 3000;
	char bufferAux[tamanhoMaximoLinha];
	char linhaCopiada;

	while (!feof(input)){
		fgets(bufferAux, tamanhoMaximoLinha, input);
		if (strstr(bufferAux, "@data") == NULL)
			fprintf(dados, bufferAux);
		else
			break;
	}

	//Insere a palavra-chave "@data" no fim do cabeçalho, de modo a indicar o início dos dados de exemplo.
	fprintf(dados, "@data\n");

	//Fecha arquivo da partição copiada.
	fclose(input);

	//Percorre a pasta de cada partição e a copia para o arquivo final.
	const int maximoParticoes = 10;
	for (int i = 0; i < maximoParticoes; i++){
		if (i != numero){

			//Constroi nome completo do diretório da partição atualmente consultada.
			sprintf(nomeArquivo, "bases/%s/it%d/%s_data.arff", nomeBase, i, nomeBase);
			input = fopen(nomeArquivo, "r");

			//Informa possível falha na abertura da partição. 
			if (input == NULL){
				printf("Arquivo não encontrado!\n");
				printf("\n%s", nomeArquivo);
				printf("\nlenght = %d", strlen(nomeArquivo));
				system("pause");
				exit(1);
			}

			//Posiciona arquivo da partição na linha de início dos dados.
			while (!feof(input)){
				fgets(bufferAux, tamanhoMaximoLinha, input);
				if (strstr(bufferAux, "@data") != NULL)
					break;
			}

			//Copia cada linha de exemplo do arquivo da partição 
			while ((linhaCopiada = fgetc(input)) != EOF){
				fputc(linhaCopiada, dados);
			}

			//Insere uma quebra de linha no arquivo final e, por fim, fecha o arquivo da partição copiada.
			fprintf(dados, "\n");
			fclose(input);
		}
	}

	//fecha arquivo de dados final.
	fclose(dados);

	//abre arquivo de dados acima já em modo de leitura.
	dados = fopen("conjunto_particoes_dados.txt", "r");
	if (dados == NULL){

		printf("Nao foi possivel criar arquivo!\n");
		printf("\n%s", "conjunto_particoes_dados.txt");
		system("pause");
		exit(1);
	}

	return dados;
}

/* Função para carrear o arquivo .arff de teste, dentre os arquivos particionados.
Considera-se que existam 10 partições, sendo o parâmetro número referente
à partição que será utilizada apenas para o teste. */
FILE* carregarArqTeste(char nomeBase[], int numero){

	//Constroi nome completo do diretório da partição de teste.
	char nomeArquivo[40];
	sprintf(nomeArquivo, "bases/%s/it%d/%s_data.arff", nomeBase, numero, nomeBase);

	//Abre arquivo da partição de teste.
	FILE* teste = fopen(nomeArquivo, "r");

	//Informa possível falha na abertura da partição. 
	if (teste == NULL){
		printf("Arquivo não encontrado!\n");
		printf("\n%s", nomeArquivo);
		printf("\nlenght = %d", strlen(nomeArquivo));
		system("pause");
		exit(1);
	}

	return teste;
}

/* Função para contar atributos da classe */
int contaAtributos(FILE* input){

	int quant_atrib = 0;																										// quantidade de atributos dos elementos da base de dados;

	while (!feof(input)){

		const int tam_str_aux = 1000;																							// define o tamanho máximo de uma linha a ser lida;	
		char str_aux[tam_str_aux];																								// string auxiliar que receberá a linha do arquivo .arff a ser lida;
		fgets(str_aux, tam_str_aux, input);																						// lê linha do arquivo;

		char* pont = strstr(str_aux, "@attribute");																				// busca pela primeira ocorrência da palavra "@attribute " na linha;

		if (pont != NULL){																										// caso a palavra seja encontrada
			quant_atrib++;																										// incrementa número de atributos;
		}
	}

	rewind(input);																												// reposiciona ponteiro no início do arquivo;
	return quant_atrib;
}

/* Função que gera uma nova estrutura "atributo" já inicializada */
atributo inicializaAtributo(){

	atributo atr;
	atr.quant_real = 0;
	atr.numerico = 0;
	atr.quant_real = 0;

	return atr;
}

void setNomeAtributo(char* string, atributo* atr){

	char* str = strchr(string, ' ');
	str = str + 1;					// desloca um caracter à direita
	int cont = 0;

	if (str != NULL){

		while (str[cont] != ' ')
			cont++;

		(*atr).nome = (char*)calloc(cont + 1, sizeof(char));
		strncpy((*atr).nome, str, cont);
		(*atr).nome[cont] = '\0';
	}
}

/* Mapeia os valores de um atributo */
atributo mapeiaAtributo(char* string){

	atributo atr = inicializaAtributo();

	char* num = strstr(string, "numeric");

	if (num != NULL){																											// caso encontre-se a palavra reservada "numeric"
		atr.numerico = 1;																										// o atributo é marcado como numérico;
	}
	else{

		setNomeAtributo(string, &atr);
		//printf("\nAqui:%s", atr.nome);
		//printf("\nSize:%d", strlen(atr.nome));
		char* str = strchr(string, '{');																						// posiciona ponteiro na primeira ocorrência de '{';
		int indice = 1;																											// posicao do caracter acessado na string "str";
		int cont_valor = 0;																										// representa a contagem dos valores que foram mapeados;
		//printf("lenght = %d\n", strlen(str));
		while ((indice + 1) < (strlen(str) - 1)){																				// enquanto o último caracter ('}') não for lido;

			int indice_valor = 0;																								// representa a posição do caractere do valor;
			while ((str[indice] != ',') && ((indice + 1) < (strlen(str) - 1))){
				//printf("(indice + 1) = %d ", (indice + 1));
				//printf("atr.valor[%d][%d] = %c\n", cont_valor, indice_valor,str[indice]);
				atr.valor[cont_valor][indice_valor] = str[indice];
				//printf("atr.quant_real = %d\n", atr.quant_real);
				indice_valor++;
				indice++;
				if (str[indice] == ',' || ((indice + 1) >= (strlen(str) - 1)))	//antes de passar para próximo valor
					atr.valor[cont_valor][indice_valor] = '\0';	//marca fim da string do nome do valor atual
			}

			atr.cod[cont_valor] = atr.quant_real;
			atr.quant_real++;
			//printf("\n");
			indice = indice + 1;
			cont_valor++;
		}

		atr.quant_real = cont_valor;
	}

	free(num);
	return atr;
}

/* Preenche o vetor de atributos da classe */
void processaAtributos(atributo* atrib, int quant_atrib, FILE* input){

	int cont = quant_atrib;																										// define contador a ser decrementado;
	const int tam_str_aux = 1000;																								// define o tamanho máximo de uma linha a ser lida;

	while (!feof(input)){

		char str_aux[tam_str_aux];																								// string auxiliar que receberá a linha do arquivo .arff a ser lida;
		fgets(str_aux, tam_str_aux, input);																						// lê linha do arquivo;

		char* pont = strstr(str_aux, "@attribute");																				// busca pela primeira ocorrência da palavra "@attribute " na linha;

		if (pont != NULL){																										// caso a palavra seja encontrada
			*(atrib + quant_atrib - cont) = mapeiaAtributo(pont);																// insere no vetor um novo atributo mapeado;
			cont--;																												// decrementa contador;
		}
	}

	rewind(input);
}

/* Imprime atributos e todos os seus possíveis valores */
void imprimeAtributos(atributo* atrib, int quant_atrib){

	printf("\n\n\t\t\t\t\t\t\t#LISTA DE ATRIBUTOS#\n\n");
	int i;
	for (i = 0; i < quant_atrib; i++){

		printf("-> Atributo %d: %s\n\n", i, (*(atrib + i)).nome);
		if ((*(atrib + i)).numerico == 0){
			int j;
			for (j = 0; j < (*(atrib + i)).quant_real; j++){
				printf("\t.Valor %2d: = %30s\t\t\t\t Mapeado para: %d\n", j, (*(atrib + i)).valor[j], (*(atrib + i)).cod[j]);
			}
		}
		else{
			printf("\t. NUMERICO\n", i);
		}
		printf("\n");
		printf("*--------------------------------------------------------------------------------------------------------*");
		printf("\n");
	}
}

char* getNomeValorAtributo(atributo* atrib, int quant_atrib, int indice_atrib, int indice_valor){

	if ((indice_atrib < quant_atrib) && (indice_valor < atrib[indice_atrib].quant_real)){

		if (indice_valor == -1)
			return "___";
		else
			return atrib[indice_atrib].valor[indice_valor];
	}
	else if (indice_atrib >= quant_atrib){

		printf("\nIndice de atributo invalido!");
		system("pause");
		exit(1);
	}
	else if (indice_valor >= atrib[indice_atrib].quant_real){

		printf("\nIndice de valor invalido!");
		printf("indice de valor = %d", indice_valor);
		system("pause");
		exit(1);
	}
}

/* Conta o número de exemplos da base de dados */
int contaExemplos(FILE* input){

	const int tam_str_aux = 1000;																									// define o tamanho máximo de uma linha a ser lida;
	char* pont;
	int contagem = 1;

	do{
		char str_aux[tam_str_aux];																								// string auxiliar que receberá a linha do arquivo .arff a ser lida;
		fgets(str_aux, tam_str_aux, input);																						// lê linha do arquivo;
		pont = strstr(str_aux, "@data");																						// busca pela primeira ocorrência da palavra "@attribute " na linha;

	} while (pont == NULL && (!feof(input)));

	char str_aux[tam_str_aux];																									// string auxiliar que receberá a linha do arquivo;
	fgets(str_aux, tam_str_aux, input);																							// lê linha do arquivo e pula para a próxima;	

	while (!feof(input)){

		char str_aux[tam_str_aux];																								// string auxiliar que receberá a linha do arquivo;
		fgets(str_aux, tam_str_aux, input);																						// lê linha do arquivo e pula para a próxima;																				// pula para a próxima linha;

		contagem++;
	}

	rewind(input);																												// reposiciona ponteiro no início do arquivo;
	return contagem;
}

/* Retorna um exemplo inicializado */
exemplo inicializaExemplo(int quant_atrib){

	exemplo exemp;
	exemp.quant_real = quant_atrib;
	return exemp;
}

int valorInt(atributo* atrib, int posicao, char* string){

	int igual = 0;																												// flag que indica se a string passada é igual a algum valor no vetor de atributos;
	int cod;
	int i;

	for (i = 0; i < (*(atrib + posicao)).quant_real; i++){

		if (strcmp((*(atrib + posicao)).valor[i], string) == 0){
			igual = 1;
			cod = (*(atrib + posicao)).cod[i];
		}
	}

	if (igual == 0){
		cod = -1;
		//printf("String = %s\n", string);
	}

	return cod;
}

/* Converte exemplo em vetor de inteiros */
exemplo converteExemplo(atributo* atrib, int quant_atrib, char* linha){

	exemplo exemp = inicializaExemplo(quant_atrib);																				// criação de um novo exemplo;
	int atrib_atual = 0;																										// atributo atual sendo lido;
	int cod;																													// código do atributo;
	char* pont;
	pont = strtok(linha, ",\n");

	while (pont != NULL){

		if (strcmp(pont, "") == 0 || strcmp(pont, "?") == 0){																	//se o valor do exemplo for vazio ou ? então seu código será -1;
			exemp.campos[atrib_atual] = -1;
		}
		else{
			cod = valorInt(atrib, atrib_atual, pont);
			if (cod == -1){																									// se o valor do exemplo não faz parte dos atributos o sistema é encerrado;
				printf("ERRO: Valor de exemplo inexistente!\n");
				system("pause");
				exit(0);
			}
			else
				exemp.campos[atrib_atual] = cod;
		}

		pont = strtok(NULL, ",\n");
		atrib_atual++;
	}

	free(pont);
	return exemp;
}

void processaExemplos(atributo* atrib, int quant_atrib, exemplo* exemplos, FILE* input){

	int cont = quant_atrib;																										// define contador a ser decrementado;
	const int tam_str_aux = 1000;																								// define o tamanho máximo de uma linha a ser lida;
	char* pont;

	do{
		char str_aux[tam_str_aux];																								// string auxiliar que receberá a linha do arquivo .arff a ser lida;
		fgets(str_aux, tam_str_aux, input);																						// lê linha do arquivo;
		pont = strstr(str_aux, "@data");																						// busca pela primeira ocorrência da palavra "@attribute " na linha;

	} while (pont == NULL && (!feof(input)));

	while (!feof(input)){

		char str_aux[tam_str_aux];																								// string auxiliar que receberá a linha do arquivo .arff a ser lida;
		fgets(str_aux, tam_str_aux, input);																						// lê linha do arquivo;
		char* pont = strstr(str_aux, "");
		*(exemplos + quant_atrib - cont) = converteExemplo(atrib, quant_atrib, pont);
		cont--;
	}
}

void imprimeExemplosInt(exemplo* exemplos, int quant_exemp){

	printf("\n\n\t\t\t\t\t\t#LISTA DOS VETORES DE EXEMPLOS#\n\n");

	int i, j;
	for (i = 0; i < quant_exemp; i++){
		printf("Exemplo %2d: ", i);

		for (j = 0; j < (*(exemplos + i)).quant_real - 1; j++){
			printf("%d, ", (*(exemplos + i)).campos[j]);
		}

		printf("%d\n\n", (*(exemplos + i)).campos[(*(exemplos + i)).quant_real - 1]);
	}

	printf("*--------------------------------------------------------------------------------------------------------*");
	printf("\n\n\n");
}

regra inicializaRegra(){

	regra r;
	int i;

	for (i = 0; i < quant_mat_cont; i++)																						// inicializa com 0's a matriz de contigência;
		r.mat_cont[i] = 0;

	r.nula = 1;																												// regra inicialmete é nula;
	r.crowding_distance = 0;
	r.quant_dominadores = 0;

	return r;
}

/* Gera uma regra aleatória */
regra geraRegra(parametros param, atributo* atrib, int quant_atrib, int classe){

	regra r = inicializaRegra();
	int i, limiar;

	//srand( (unsigned) time(NULL) );
	for (i = 0; i < quant_atrib - 1; i++){

		if (param.metodo_gerar_regras == 1){

			limiar = rand() % 101;
			if (param.prob_valor_vazio >= limiar)
				r.valores[i] = -1;
			else
				r.valores[i] = rand() % ((*(atrib + i)).quant_real);
		}
		else if (param.metodo_gerar_regras == 0)
			r.valores[i] = rand() % ((*(atrib + i)).quant_real + 1) - 1;
	}

	switch (classe){

	case 0:
		r.valores[quant_atrib - 1] = 0;
		break;

	case 1:
		r.valores[quant_atrib - 1] = 1;
		break;

	default:
		r.valores[quant_atrib - 1] = rand() % ((*(atrib + i)).quant_real);												// valor aleatório para a classe;
	}

	r.nula = 0;																												// regra deixa de ser nula;

	return r;
}

void imprimeRegra(regra r, int quant_atrib){

	//printf("\nRegra: ");

	int i;
	for (i = 0; i < quant_atrib - 1; i++){

		printf("%d ", r.valores[i]);
	}

	printf("-> %d\n", r.valores[quant_atrib - 1]);
}

void imprimeRegraCompleta(regra r, int quant_atrib){

	//printf("\nRegra: ");

	int i;
	for (i = 0; i < quant_atrib - 1; i++){

		printf("%d ", r.valores[i]);
	}

	printf("-> %d\n", r.valores[quant_atrib - 1]);

	printf("\n\tCrowding Distance: %f", r.crowding_distance);
	printf("\n\tQuantidade de dominadores: %d\n\n", r.quant_dominadores);
}

void imprimeNomesRegra(regra r, atributo* atrib, int quant_atrib){

	int i;
	for (i = 0; i < quant_atrib - 1; i++){

		printf("%s ", getNomeValorAtributo(atrib, quant_atrib, i, r.valores[i]));
	}

	printf("-> %s\n", getNomeValorAtributo(atrib, quant_atrib, quant_atrib - 1, r.valores[quant_atrib - 1]));
}

/* Zera todas as posições da matriz de contingência de uma regra */
void zeraMatrizCont(regra* r){

	int i;

	for (i = 0; i < quant_mat_cont; i++){

		(*r).mat_cont[i] = 0;
	}
}

void preencheMatrizCont(regra* r, exemplo* exemplos, int quant_exemp, int quant_atrib){

	zeraMatrizCont(r);
	int i, j, b, h;

	for (i = 0; i < quant_exemp; i++){

		b = 1;
		h = 1;

		for (j = 0; j < quant_atrib - 1; j++){

			if (((*(exemplos + i)).campos[j] != (*r).valores[j]) && ((*r).valores[j] != -1)){
				b = 0;
			}
		}

		if ((*(exemplos + i)).campos[quant_atrib - 1] != (*r).valores[quant_atrib - 1]){
			h = 0;
		}

		if (b == 1)
			(*r).mat_cont[B]++;
		else
			(*r).mat_cont[_B]++;

		if (h == 1)
			(*r).mat_cont[H]++;
		else
			(*r).mat_cont[_H]++;

		if (b == 1 && h == 1)
			(*r).mat_cont[BH]++;

		else if (b == 1 && h == 0)
			(*r).mat_cont[B_H]++;

		else if (b == 0 && h == 1)
			(*r).mat_cont[_BH]++;

		else if (b == 0 && h == 0)
			(*r).mat_cont[_B_H]++;
	}
}

void imprimeMatrizCont(regra r){

	printf("\n-> Matriz de Contingencia:\n");
	printf("\nBH = %d%20s", r.mat_cont[BH], " ");
	printf("_BH = %d%20s", r.mat_cont[_BH], " ");
	printf("B_H = %d\n", r.mat_cont[B_H]);
	printf("_B_H = %d%20s", r.mat_cont[_B_H], " ");
	printf("B = %d\%20s", r.mat_cont[B], " ");
	printf("_B = %d\n", r.mat_cont[_B]);
	printf("H = %d%20s", r.mat_cont[H], " ");
	printf("_H = %d\n", r.mat_cont[_H]);
}

void calculaFuncoesObj(regra* r, int quant_atrib, atributo* atrib, int quant_exemp){

	int quant_classes = (*(atrib + quant_atrib - 1)).quant_real;

	if ((*r).mat_cont[B] != 0)
		(*r).func_ob[ACC] = (double)((*r).mat_cont[BH]) / (*r).mat_cont[B];
	else
		(*r).func_ob[ACC] = -1;
	if ((*r).mat_cont[B] != 0)
		(*r).func_ob[ERR] = (double)((*r).mat_cont[B_H]) / (*r).mat_cont[B];
	else
		(*r).func_ob[ERR] = -1;
	if ((*r).mat_cont[_B] != 0)
		(*r).func_ob[NEGREL] = (double)((*r).mat_cont[_B_H]) / (*r).mat_cont[_B];
	else
		(*r).func_ob[NEGREL] = -1;
	if (((*r).mat_cont[B] + quant_classes) != 0)
		(*r).func_ob[ACCLP] = (double)((*r).mat_cont[BH] + 1) / ((*r).mat_cont[B] + quant_classes);
	else
		(*r).func_ob[ACCLP] = -1;
	if ((*r).mat_cont[H] != 0)
		(*r).func_ob[SENS] = (double)((*r).mat_cont[BH]) / (*r).mat_cont[H];
	else
		(*r).func_ob[SENS] = -1;
	if ((*r).mat_cont[_H] != 0)
		(*r).func_ob[SPEC] = (double)((*r).mat_cont[_B_H]) / (*r).mat_cont[_H];
	else
		(*r).func_ob[SPEC] = -1;
	if (quant_exemp != 0)
		(*r).func_ob[COV] = (double)((*r).mat_cont[B]) / quant_exemp;
	else
		(*r).func_ob[COV] = -1;
	if (quant_exemp != 0)
		(*r).func_ob[SUP] = (double)((*r).mat_cont[BH]) / quant_exemp;
	else
		(*r).func_ob[SUP] = -1;
}

void imprimeFuncoesObj(regra r){

	printf("\n-> Funcoes Objetivo:\n");
	printf("\n%30s %.5f", "Precisao = ", r.func_ob[ACC]);
	printf("\n%30s %.5f", "Erro = ", r.func_ob[ERR]);
	printf("\n%30s %.5f", "Confianca negativa = ", r.func_ob[NEGREL]);
	printf("\n%30s %.5f", "Precisao de Laplace = ", r.func_ob[ACCLP]);
	printf("\n%30s %.5f", "Sensitividade = ", r.func_ob[SENS]);
	printf("\n%30s %.5f", "Especificidade = ", r.func_ob[SPEC]);
	printf("\n%30s %.5f", "Cobertura = ", r.func_ob[COV]);
	printf("\n%30s %.5f", "Suporte = ", r.func_ob[SUP]);
}

void imprimeFuncaoObj(regra r, int indice){

	switch (indice){

	case ACC:
		printf("\n%30s %.5f", "Precisao = ", r.func_ob[ACC]);
		break;
	case ERR:
		printf("\n%30s %.5f", "Erro = ", r.func_ob[ERR]);
		break;
	case NEGREL:
		printf("\n%30s %.5f", "Confianca negativa = ", r.func_ob[NEGREL]);
		break;
	case ACCLP:
		printf("\n%30s %.5f", "Precisao de Laplace = ", r.func_ob[ACCLP]);
		break;
	case SENS:
		printf("\n%30s %.5f", "Sensitividade = ", r.func_ob[SENS]);
		break;
	case SPEC:
		printf("\n%30s %.5f", "Especificidade = ", r.func_ob[SPEC]);
		break;
	case COV:
		printf("\n%30s %.5f", "Cobertura = ", r.func_ob[COV]);
		break;
	case SUP:
		printf("\n%30s %.5f", "Suporte = ", r.func_ob[SUP]);
		break;
	}
}

/* regra* geraRegras(parametros param, int quant_regras, atributo* atrib, int quant_atrib, int classe, exemplo* exemplos, int quant_exemp){

int i;
regra* regras = (regra*)calloc(quant_regras, sizeof(regra));
if (regras == NULL){

printf("\nErro de alcocacao!");
system("pause");
exit(1);
}

for (i = 0; i < quant_regras; i++){

*(regras + i) = geraRegra(param, atrib, quant_atrib, classe);
preencheMatrizCont((regras + i), exemplos, quant_exemp, quant_atrib);
calculaFuncoesObj((regras + i), quant_atrib, atrib, quant_exemp);
//imprimeRegra(*(regras + i), quant_atrib);
//imprimeMatrizCont(*(regras + i));
//imprimeFuncoesObj(*(regras + i));
}

return regras;
} */

/* Dado uma string, a função retorna o restante da linha de um arquivo após tal string, somente quando a linha começa pela string*/
char* saltaStringArq(char* string, FILE* arq){

	const int tam_str_aux = 200;																							// define o tamanho máximo de uma linha a ser lida;
	char str_aux[tam_str_aux];																								// string auxiliar que receberá a linha do arquivo .arff a ser lida;

	fgets(str_aux, tam_str_aux, arq);																						// lê linha do arquivo;
	char* pont = (char*)calloc(tam_str_aux, sizeof(char));																	// aloca espaço de memória equivalente ao tamanho máximo da linha lida;
	if (pont == NULL){

		printf("\nErro de alcocacao!");
		system("pause");
		exit(1);
	}

	strncpy(pont, str_aux, tam_str_aux);																					// copia a linha lida para o ponteiro "pont";
	pont = (pont + strlen(string));																							// armazena no ponteiro "pont" o resto da linha após "string";
	return pont;
}

/* Dado uma string, a função retorna o restante da linha de um arquivo após tal string, independentemente da posição da string*/
char* localizaString(char* string, FILE* arq){

	const int tam_str_aux = 200;																							// define o tamanho máximo de uma linha a ser lida;
	char str_aux[tam_str_aux];																								// string auxiliar que receberá a linha do arquivo .arff a ser lida;

	fgets(str_aux, tam_str_aux, arq);																						// lê linha do arquivo;
	char* pont = (char*)calloc(tam_str_aux, sizeof(char));																	// aloca espaço de memória equivalente ao tamanho máximo da linha lida;
	if (pont == NULL){

		printf("\nErro de alcocacao!");
		system("pause");
		exit(1);
	}

	strncpy(pont, str_aux, tam_str_aux);																					// copia a linha lida para o ponteiro "pont";

	int indice_linha = 0, indice_string = 0;
	int flag = 0;

	while (indice_linha < strlen(pont) && flag == 0){
		while (indice_string < strlen(string) && indice_linha + indice_string < strlen(pont) & flag == 0){
			if (string[indice_string] == pont[indice_linha + indice_string]){
				indice_string++;
				if (indice_string == strlen(string))
					flag = 1;
			}
			else{
				indice_linha++;
				indice_string = 0;
			}
		}
	}

	if (flag == 1){
		pont = (pont + indice_linha + strlen(string));
	}
	else
		return NULL;

	return pont;
}

void carregaParametros(FILE* file, parametros* param){

	(*param).arquivo = strtok(saltaStringArq("@arquivo_treino:", file), "\n");
	(*param).execucoes = atoi(saltaStringArq("@execucoes:", file));
	(*param).classe = atoi(saltaStringArq("@classe:", file));
	(*param).funcao_obj = atoi(saltaStringArq("@funcao_objetivo:", file));
	(*param).bl_interacoes = atoi(saltaStringArq("@num_inter_bl:", file));
	(*param).bl_vizinhos = atoi(saltaStringArq("@num_vizinhos_bl:", file));
	(*param).funcoes_obj_pareto = strtok(saltaStringArq("@dominancia_de_pareto:", file), "\n");
	(*param).quant_regras_pareto = atoi(saltaStringArq("@quant_regras_pareto:", file));
	(*param).quant_particulas = atoi(saltaStringArq("@quant_particulas:", file));
	(*param).quant_enxames = atoi(saltaStringArq("@quant_enxames:", file));
	(*param).tamanho_arquivo = atoi(saltaStringArq("@tamanho_arquivo:", file));
	(*param).metodo_dopagem_solucao = atoi(saltaStringArq("@metodo_dopagem_solucao:", file));
	(*param).metodo_gerar_regras = atoi(saltaStringArq("@metodo_gerar_regras:", file));
	(*param).prob_valor_vazio = atoi(saltaStringArq("@prob_valor_vazio:", file));

	rewind(file);
}

void imprimeParametros(parametros param){

	printf("\nParametros:\n\n");
	printf("Arquivo de treino = %s\n", param.arquivo);
	printf("Execucoes = %d\n", param.execucoes);
	printf("Classe = %d\n", param.classe);
	printf("Funcao Objetivo = %d\n", param.funcao_obj);
	printf("Numero de interacoes = %d\n", param.bl_interacoes);
	printf("Numero de vizinhos = %d\n", param.bl_vizinhos);
	printf("Funcoes objetivo de Pareto = %s\n", param.funcoes_obj_pareto);
	printf("Quantidade de regras de Pareto = %d\n", param.quant_regras_pareto);
	printf("Quantidade de particulas = %d\n", param.quant_particulas);
	printf("Quantidade de enxames = %d\n", param.quant_enxames);
	printf("Tamanho maximo do arquivo = %d\n", param.tamanho_arquivo);
	printf("Método de dopagem de solucao = %d\n", param.metodo_dopagem_solucao);
	printf("Método de geração de regras = %d\n", param.metodo_gerar_regras);
	printf("Probabilidade de gerar atributo vazio (_) = %d\n", param.prob_valor_vazio);
}

/* Altera aleatoriamente um atributo de uma certa regra passada como parâmetro */
regra alteraRegra(regra base, atributo* atrib, int quant_atrib){

	regra nova = base;
	int indice = rand() % (quant_atrib - 1);																					// indice aleatório do atributo a ser alterado;
	nova.valores[indice] = rand() % ((*(atrib + indice)).quant_real + 1) - 1;

	printf("\n\nBase = :");
	imprimeRegra(base, quant_atrib);
	printf("\n\nNova = :");
	imprimeRegra(nova, quant_atrib);

	return nova;
}

/* Função que retorna 0 caso a 1ª regra seja melhor ou igual que a 2ª regra  */
int comparaRegras(regra r0, regra r1, int func_ob){

	if (func_ob == ERR){
		if (r0.func_ob[ERR] <= r1.func_ob[ERR])
			return 0;
		else
			return 1;
	}
	else{
		if (r0.func_ob[func_ob] >= r1.func_ob[func_ob])
			return 0;
		else
			return 1;
	}
}

//////////////////////////////////////// INÍCIO DOMINÂNCIA DE PARETO //////////////////////////

/*Retorna -1 se r1 domina r2, 0 se nenhuma das duas regras domina a outra, 1 se r2 domina r1*/
int dominaPorPareto(regra r1, regra r2, regiao_pareto pareto){

	int i;
	int saida = 0;

	for (i = 0; i < quant_func_ob; i++){
		if (pareto.func_obj[i] == 1){
			if (r1.func_ob[i] > r2.func_ob[i]){

				if (saida == 0 || saida == -1)
					saida = -1;
				else
					return 0;
			}
			if (r1.func_ob[i] < r2.func_ob[i]){

				if (saida == 0 || saida == 1)
					saida = 1;
				else
					return 0;
			}
		}
	}

	return saida;
}

/*Cria e inicializa uma variável do tipo "regiao_pareto"*/
regiao_pareto inicializaPareto(parametros param){

	regiao_pareto pareto;

	pareto.solucoes = (regra*)calloc(1, sizeof(regra));
	if (pareto.solucoes == NULL){

		printf("\nErro de alcocacao!");
		system("pause");
		exit(1);
	}

	pareto.solucoes[0].nula = 1;
	pareto.quant_sol_pareto = 1;

	int i;
	for (i = 0; i < quant_func_ob; i++){
		pareto.func_obj[i] = (param.funcoes_obj_pareto[i]) - 48;
	}

	return pareto;
}

void inserePareto(regiao_pareto* pareto, regra r){
	int i, inserido = 0;

	for (i = 0; i < (*pareto).quant_sol_pareto; i++){

		if ((*pareto).solucoes[i].nula == 1){

			(*pareto).solucoes[i] = r;
			(*pareto).solucoes[i].nula = 0;
			inserido = 1;
			break;
		}
	}

	if (inserido == 0){

		(*pareto).solucoes = (regra*)realloc((*pareto).solucoes, ((*pareto).quant_sol_pareto + 1)*sizeof(regra));
		(*pareto).quant_sol_pareto++;
		(*pareto).solucoes[(*pareto).quant_sol_pareto - 1] = r;
		(*pareto).solucoes[(*pareto).quant_sol_pareto - 1].nula = 0;
	}
}

int contaRegrasNaoNulasPareto(regiao_pareto pareto){

	int cont = 0;
	int i;

	for (i = 0; i < pareto.quant_sol_pareto; i++){

		if (pareto.solucoes[i].nula == 0)
			cont++;
	}

	return cont;
}

void apagaSolucoesNulasPareto(regiao_pareto* pareto){

	int quant_sol_nao_nulas = contaRegrasNaoNulasPareto(*pareto);
	regra* solucoes = (regra*)calloc(quant_sol_nao_nulas, sizeof(regra));
	if (solucoes == NULL){

		printf("\nErro de alcocacao!");
		system("pause");
		exit(1);
	}

	int cont = 0;
	int i;

	for (i = 0; i < (*pareto).quant_sol_pareto; i++){

		if ((*pareto).solucoes[i].nula == 0){

			solucoes[cont] = (*pareto).solucoes[i];
			cont++;
		}
	}

	(*pareto).solucoes = solucoes;
	(*pareto).quant_sol_pareto = quant_sol_nao_nulas;
}

void imprimeDominioPareto(regiao_pareto pareto, int quant_atrib){

	printf("\n#Dominio de Pareto:\n");

	int i;
	for (i = 0; i < pareto.quant_sol_pareto; i++){

		if (pareto.solucoes[i].nula == 0)
			imprimeRegra(pareto.solucoes[i], quant_atrib);
	}
}

void imprimeDominioParetoComObjetivos(regiao_pareto pareto, int quant_atrib, parametros param){

	printf("\n#Dominio de Pareto:\n");

	int i, j;
	for (i = 0; i < pareto.quant_sol_pareto; i++){

		printf("\n");
		imprimeRegra(pareto.solucoes[i], quant_atrib);

		for (j = 0; j < quant_func_ob; j++){

			if (param.funcoes_obj_pareto[j] == '1'){
				imprimeFuncaoObj(pareto.solucoes[i], j);
			}
		}
	}
}

void zeraQuantDominadoresRegras(regra* regras, int quant_regras){

	int i;
	for (i = 0; i < quant_regras; i++){

		regras[i].quant_dominadores = 0;
	}
}

regiao_pareto dominanciaDePareto(parametros param, regra* regras, int quant_regras){

	regiao_pareto pareto = inicializaPareto(param);
	zeraQuantDominadoresRegras(regras, quant_regras);
	int i, j;
	//int dominada;																										// dominada = 0 se regra não é dominada por nenhuma outra regra dentro de "pareto";

	for (i = 0; i < quant_regras; i++){

		//dominada = 0;

		for (j = 0; j < pareto.quant_sol_pareto; j++){

			if (pareto.solucoes[j].nula == 0){

				if (dominaPorPareto(*(regras + i), pareto.solucoes[j], pareto) == -1){
					pareto.solucoes[j].nula = 1;
					pareto.solucoes[j].quant_dominadores++;
				}
				else if (dominaPorPareto(*(regras + i), pareto.solucoes[j], pareto) == 1){
					//dominada = 1;
					regras[i].quant_dominadores++;
					//printf("\n%d",regras[i].quant_dominadores);
				}
			}
		}

		if ( /*dominada == 0*/ regras[i].quant_dominadores == 0){

			//printf("\ninserindo em pareto: %d", regras[i].quant_dominadores);
			inserePareto(&pareto, *(regras + i));
		}
	}

	apagaSolucoesNulasPareto(&pareto);
	return pareto;
}

//////////////////////////////////////// FIM DOMINÂNCIA DE PARETO /////////////////////////////

//////////////////////////////////////// INÍCIO S-PSO ////////////////////////////////////////

/*aloca no device os espaços de memória necessários para o armazenamento da velocidade de uma partícula e
preenche tais espaços com a velocidade da partícula armazenada no host*/
double** carregaVelocidadeParticulaDevice(particula p, atributo* atrib, int quant_atrib){

	//aloca vetor vel no próprio host
	double** vel = (double**)calloc(quant_atrib, sizeof(double*));
	if (vel == NULL){

		printf("\nErro de alcocacao!");
		system("pause");
		exit(1);
	}

	//para cada ponteiro de vel, aloca espaço no device e copia valor do host
	for (int i = 0; i < quant_atrib; i++){
		cudaMalloc((void **)&vel[i], (atrib[i].quant_real + 1) * sizeof(double));
		if (vel[i] == NULL){

			printf("\nErro de alcocacao!");
			system("pause");
			exit(1);
		}
		cudaMemcpy(vel[i], p.velocidade[i], (atrib[i].quant_real + 1) * sizeof(double), cudaMemcpyHostToDevice);
	}

	//aloca vetor vel_d no device e copia de vel as referências aos ponteiros alocados no device
	double** vel_d;
	cudaMalloc(&vel_d, quant_atrib * sizeof(double));
	cudaMemcpy(vel_d, vel, quant_atrib * sizeof(double), cudaMemcpyHostToDevice);

	return vel_d;
}

double** inicializaVelocidadeParticula(atributo* atrib, int quant_atrib){

	double** vel = (double**)calloc(quant_atrib, sizeof(double*));
	if (vel == NULL){

		printf("\nErro de alcocacao!");
		system("pause");
		exit(1);
	}

	int i, j;
	for (i = 0; i < quant_atrib; i++){

		vel[i] = (double*)calloc(atrib[i].quant_real + 1, sizeof(double));
		if (vel[i] == NULL){

			printf("\nErro de alcocacao!");
			system("pause");
			exit(1);
		}
		for (j = 0; j < atrib[i].quant_real + 1; j++){

			vel[i][j] = (rand() % 101) / 100.0;
		}
	}

	return vel;
}

particula criaParticula(parametros param, atributo* atrib, int quant_atrib){

	particula p;
	p.posicao = geraRegra(param, atrib, quant_atrib, param.classe);
	//imprimeRegra(p.posicao, quant_atrib);
	p.lBest = p.posicao;
	p.gBest = p.posicao;
	p.gBest.nula = 1;
	p.velocidade = inicializaVelocidadeParticula(atrib, quant_atrib);

	return p;
}

particula* criaEnxame(parametros param, atributo* atrib, int quant_atrib){

	particula* enxame = (particula*)calloc(param.quant_particulas, sizeof(particula));
	if (enxame == NULL){

		printf("\nErro de alcocacao!");
		system("pause");
		exit(1);
	}

	int i;
	for (i = 0; i < param.quant_particulas; i++){

		enxame[i] = criaParticula(param, atrib, quant_atrib);

	}

	return enxame;
}

void calculaObjetivosEnxame(particula* enxame, parametros param, exemplo* exemplos, int quant_exemp, atributo* atrib, int quant_atrib){

	int i;
	for (i = 0; i < param.quant_particulas; i++){
		regra r = enxame[i].posicao;
		preencheMatrizCont(&(enxame[i].posicao), exemplos, quant_exemp, quant_atrib);
		//imprimeRegra(enxame[i].posicao, quant_atrib);
		calculaFuncoesObj(&(enxame[i].posicao), quant_atrib, atrib, quant_exemp);
		//enxame[i].posicao = r;
		//imprimeRegra(enxame[i].posicao, quant_atrib);
	}
}


void imprimeVelocidadeParticula(particula p, atributo* atrib, int quant_atrib){

	printf(".Velocidade:\n");

	int i, j;
	for (i = 0; i < quant_atrib; i++){

		printf("\n\tatributo %d", i + 1);
		for (j = 0; j < atrib[i].quant_real + 1; j++){

			printf("\n\t%20s - %.2f%%", getNomeValorAtributo(atrib, quant_atrib, i, j - 1), p.velocidade[i][j]);
		}
	}
}

void imprimeParticula(particula p, parametros param, atributo* atrib, int quant_atrib){

	if (p.posicao.nula == 0){

		printf("\n.Posicao = ");
		imprimeRegraCompleta(p.posicao, quant_atrib);
	}

	if (p.lBest.nula == 0){

		printf(".Melhor local = ");
		imprimeRegraCompleta(p.lBest, quant_atrib);
	}

	if (p.gBest.nula == 0){

		printf(".Melhor global = ");
		imprimeRegraCompleta(p.gBest, quant_atrib);
	}

	imprimeVelocidadeParticula(p, atrib, quant_atrib);

	printf("\n.Objetivos:");

	int i;
	for (i = 0; i < quant_func_ob; i++){

		if (param.funcoes_obj_pareto[i] == '1'){
			imprimeFuncaoObj(p.posicao, i);
		}
	}
}

void imprimeEnxame(particula* enxame, parametros param, atributo* atrib, int quant_atrib){

	printf("\n#Enxame:\n");

	int i;
	for (i = 0; i < param.quant_particulas; i++){

		printf("\nParticula %d:\n", i + 1);
		imprimeParticula(enxame[i], param, atrib, quant_atrib);
	}
}

regra* enxameParaRegras(parametros param, particula* enxame){

	regra* regras = (regra*)calloc(param.quant_particulas, sizeof(regra));
	if (regras == NULL){

		printf("\nErro de alcocacao!");
		system("pause");
		exit(1);
	}

	int i;
	for (i = 0; i < param.quant_particulas; i++){

		regras[i] = enxame[i].posicao;
	}

	return regras;
}

/*Dado dois vetores de regras v1 e v2 e seus respectivos tamanhos t1 e t2,
retorna um novo vetor v de tamanho t1 + t2 contendo as regras de v1 e v2*/
regra* uneRegras(regra* regras1, int quant_regras1, regra* regras2, int quant_regras2){

	regra* regras = (regra*)calloc(quant_regras1 + quant_regras2, sizeof(regra));
	if (regras == NULL){

		printf("\nErro de alcocacao!");
		system("pause");
		exit(1);
	}

	int i;
	for (i = 0; i < quant_regras1; i++){

		if (regras1[i].nula != 1)
			regras[i] = regras1[i];
	}
	for (i = 0; i < quant_regras2; i++){

		if (regras2[i].nula != 1)
			regras[i + quant_regras1] = regras2[i];
	}

	return regras;
}

int verificaIgualdadeRegras(regra r1, regra r2, int quant_atrib){

	int iguais = 1;
	int i;

	for (i = 0; i < quant_atrib; i++){

		if (r1.valores[i] != r2.valores[i])
			iguais = 0;
	}

	return iguais;
}

regra* apagaRegrasIguais(regra* r, int* quant_regras, int quant_atrib){

	int i, j;

	for (i = 0; i < *quant_regras; i++){
		for (j = 0; j < *quant_regras; j++){

			if ((i != j) && (r[i].nula == 0) && (r[j].nula == 0) && (verificaIgualdadeRegras(r[i], r[j], quant_atrib) == 1)){

				r[i].nula = 1;
			}
		}
	}

	int cont = 0;

	for (i = 0; i < *quant_regras; i++){

		if (r[i].nula == 0)
			cont++;
	}
	//printf("\n%d %d", *quant_regras, cont);
	regra* regras = (regra*)calloc(cont, sizeof(regra));
	if (regras == NULL){

		printf("\nErro de alcocacao!");
		system("pause");
		exit(1);
	}

	cont = 0;

	for (i = 0; i < *quant_regras; i++){

		if (r[i].nula == 0){

			regras[cont] = r[i];
			cont++;
		}
	}

	*quant_regras = cont;

	return regras;
}

void setLBest(parametros param, regiao_pareto pareto, particula* enxame){

	int i;
	for (i = 0; i < param.quant_particulas; i++){

		if (dominaPorPareto(enxame[i].posicao, enxame[i].lBest, pareto) == -1)
			enxame[i].lBest = enxame[i].posicao;
	}
}

__device__ __host__ int comparaSENS(const void *a, const void *b){
	regra *x = (regra *)a;
	regra *y = (regra *)b;

	if ((*x).func_ob[SENS] > (*y).func_ob[SENS])
		return 1;
	if ((*x).func_ob[SENS] < (*y).func_ob[SENS])
		return -1;
	return 0;
}

__device__ __host__ int comparaSPEC(const void *a, const void *b){
	regra *x = (regra *)a;
	regra *y = (regra *)b;

	if ((*x).func_ob[SPEC] > (*y).func_ob[SPEC])
		return 1;
	if ((*x).func_ob[SPEC] < (*y).func_ob[SPEC])
		return -1;
	return 0;
}

void atualizaCrowdingDistance(regiao_pareto pareto, int objetivo, regra* regras, int quant_regras){

	int i, j;
	/* imprimeDominioParetoComObjetivos(*pareto, quant_atrib, param);

	printf("\nAntigas Crowding Distances:\n");
	for(j = 0; j < (*pareto).quant_sol_pareto; j++){

	printf("%f ", (*pareto).solucoes[j].crowding_distance);
	} */

	switch (objetivo){
	case ACC:
		break;
	case ERR:
		break;
	case NEGREL:
		break;
	case ACCLP:
		break;
	case SENS:
		qsort(regras, quant_regras, sizeof(regra), comparaSENS);
		break;
	case SPEC:
		qsort(regras, quant_regras, sizeof(regra), comparaSPEC);
		break;
	case COV:
		break;
	case SUP:
		break;
	}

	for (i = 0; i < quant_regras; i++){
		if (regras[i].func_ob[objetivo] == regras[0].func_ob[objetivo] || regras[i].func_ob[objetivo] == regras[quant_regras - 1].func_ob[objetivo])
			regras[i].crowding_distance = INFINITO;
	}

	//regras[0].crowding_distance = INFINITO;
	for (i = 1; i < quant_regras - 1; i++){
		if (regras[i].crowding_distance != INFINITO)
			regras[i].crowding_distance += regras[i + 1].func_ob[objetivo] - regras[i - 1].func_ob[objetivo];
	}
	//regras[quant_regras - 1].crowding_distance = INFINITO;

	/* printf("\nNovas Crowding Distances:\n");
	for(j = 0; j < (*pareto).quant_sol_pareto; j++){

	printf("%f ", (*pareto).solucoes[j].crowding_distance);
	} */
}

void atualizaCrowdingDistances(regiao_pareto pareto, regra* regras, int quant_regras){

	int i;
	for (i = 0; i < quant_func_ob; i++){

		if (pareto.func_obj[i] == 1)
			atualizaCrowdingDistance(pareto, i, regras, quant_regras);
	}

}

int comparaDominadores(const void *a, const void *b){
	regra *x = (regra *)a;
	regra *y = (regra *)b;

	if ((*x).quant_dominadores > (*y).quant_dominadores)
		return 1;
	if ((*x).quant_dominadores < (*y).quant_dominadores)
		return -1;
	return 0;
}

void ordenaRegrasMenosDominadas(regra* regras, int quant_regras){
	qsort(regras, quant_regras, sizeof(regra), comparaDominadores);
}

int comparaCrowdDistances(const void *a, const void *b){
	regra *x = (regra *)a;
	regra *y = (regra *)b;

	if ((*x).crowding_distance > (*y).crowding_distance)
		return -1;
	if ((*x).crowding_distance < (*y).crowding_distance)
		return 1;
	return 0;
}

void ordenaRegrasMenoresCrowdDistances(regra* regras, int quant_regras){
	qsort(regras, quant_regras, sizeof(regra), comparaCrowdDistances);
}

void setGBest(parametros param, regiao_pareto pareto, particula* enxame){

	int i;
	regra r1, r2;

	for (i = 0; i < param.quant_particulas; i++){

		r1 = pareto.solucoes[rand() % pareto.quant_sol_pareto];
		r2 = pareto.solucoes[rand() % pareto.quant_sol_pareto];

		if (r1.crowding_distance > r2.crowding_distance)
			enxame[i].gBest = r1;
		else
			enxame[i].gBest = r2;
	}
}
/*
void setGBest(parametros param, regiao_pareto pareto, particula* enxame){

int i;
for(i = 0; i < param.quant_particulas; i++){

enxame[i].gBest = pareto.solucoes[rand()%pareto.quant_sol_pareto];
}
}
*/
double getOmega(){

	double omega;

	/* do{
	omega = (rand()%101)/100.0;
	}while(omega > MAX_OMEGA); */

	return MAX_OMEGA; //omega;
}

double getPhi(){

	double phi;

	do{
		phi = (rand() % 101) / 100.0;
	} while (phi > MAX_OMEGA);

	return phi;
}

double getConst(){

	return 2;//1.5 + (rand()%101)/100.0;
}

int posicaoMenosPosicao(int p1, int p2){
	//printf("\nposicao1 = %d", p1);
	//printf("\nposicao2 = %d", p2);
	if (p1 != p2)
		return p1;
	else
		return -2;	// -2 representa um valor fora do domínio dos atributo da regra;
}

double coefVezesPosicao(double c, int posicao){

	double phi = getPhi();
	//printf("\nposicao = %d", posicao);
	//printf("\nphi = %f", phi);
	//printf("\nc = %f", c);
	if (posicao == -2)
		return 0;
	else if (c*phi < 1)
		return c*phi;
	else
		return 1;
}

void coefVezesVelocidade(double* velocidade, int quant_vel){

	double coef = getOmega();
	int i;

	for (i = 0; i < quant_vel; i++){

		if (coef*velocidade[i] < 1)
			velocidade[i] = coef*velocidade[i];
		else
			velocidade[i] = 1;
	}
}

/*Dada a velocidade calculada de um valor de um atributo, verifica
se esse valor é maior que o correspondente dentro do vetor de velocidades*/
void somaVelocidades(double* velocidades, double vel, int atributo){
	//printf("\natributo: %d", atributo);
	//printf("\nvelocidades[] = %f, vel = %f", velocidades[atributo], vel);
	if (velocidades[atributo + 1] < vel)
		velocidades[atributo + 1] = vel;
}

void setVelocidadeParticula(particula* p, atributo* atrib, int quant_atrib, parametros param){

	double c1 = getConst();
	double c2 = getConst();

	int i;
	for (i = 0; i < quant_atrib; i++){

		coefVezesVelocidade((*p).velocidade[i], atrib[i].quant_real + 1);
		//printf("\naqui1:\n");
		//imprimeParticula(*p, param, atrib, quant_atrib);
		//printf("\nterminou1\n");
		somaVelocidades((*p).velocidade[i], coefVezesPosicao(c1, posicaoMenosPosicao((*p).lBest.valores[i], (*p).posicao.valores[i])), (*p).lBest.valores[i]);
		//printf("\naqui2:\n");
		//printf("i = %d", i);
		//printf("\nvalores[i] = %d", (*p).gBest.valores[i]);
		//imprimeParticula(*p, param, atrib, quant_atrib);
		//printf("\nterminou2\n");
		somaVelocidades((*p).velocidade[i], coefVezesPosicao(c2, posicaoMenosPosicao((*p).gBest.valores[i], (*p).posicao.valores[i])), (*p).gBest.valores[i]);
		//printf("\naqui3:\n");
		//imprimeParticula(*p, param, atrib, quant_atrib);
		//printf("\nterminou3\n");
	}
}

double getMaiorVelocidade(double* velocidades, int quant_vel){

	double vel = 0;
	int i;

	for (i = 0; i < quant_vel; i++){

		if (velocidades[i] > vel)
			vel = velocidades[i];
	}

	return vel;
}

double getRandFloat(double limite){

	double r;

	do{
		r = (rand() % 101) / 100.0;
	} while (r > limite);

	return r;
}

int roleta(double* velocidade, atributo atrib){

	double limite = getMaiorVelocidade(velocidade, atrib.quant_real + 1);
	double alfa = getRandFloat(limite);
	int posicao;
	//printf("\nRoleta:");
	//printf("\nlimite = %f", limite);
	//printf("\nalfa = %f", alfa);

	do{
		posicao = rand() % (atrib.quant_real + 1) - 1;
		//printf("\nposicao = %d", posicao);
		//printf("\nvelocidade = %f\n", velocidade[posicao]);
	} while (velocidade[posicao + 1] < alfa);

	return posicao;
}

void setPosicaoParticula(particula* p, atributo* atrib, int quant_atrib){

	int i;
	for (i = 0; i < quant_atrib; i++){

		(*p).posicao.valores[i] = roleta((*p).velocidade[i], atrib[i]);
	}
}

//cria o arquivo de saída contendo as soluções do algoritmo (execucao - corresponde ao número da execução do algoritmo como um todo)
FILE* criaArquivo(int execucao){

	char nome_aqr[22];
	sprintf(nome_aqr, "solucoes - %d.txt", execucao);

	FILE* output = fopen(nome_aqr, "w");

	if (output == NULL){

		printf("Nao foi possivel criar arquivo!\n");
		system("pause");
		exit(1);
	}
	else
		return output;
}

void imprimeNomesRegraEmArquivo(regra r, atributo* atrib, int quant_atrib, FILE* output){

	int i;
	for (i = 0; i < quant_atrib - 1; i++){

		fprintf(output, "%s ", getNomeValorAtributo(atrib, quant_atrib, i, r.valores[i]));
	}

	fprintf(output, "-> %s\n", getNomeValorAtributo(atrib, quant_atrib, quant_atrib - 1, r.valores[quant_atrib - 1]));
}

void insereNomesSolucoesArquivo(regiao_pareto solucao, parametros param, FILE* output, atributo* atrib, int quant_atrib){

	int i;
	for (i = 0; i < solucao.quant_sol_pareto; i++){
		imprimeNomesRegraEmArquivo(solucao.solucoes[i], atrib, quant_atrib, output);
	}
}

void insereObjEmArquivo(regiao_pareto solucao, particula* enxame, parametros param, FILE* output, int interacao){

	int i, j;

	fprintf(output, "Interação: %d\n", interacao);

	fprintf(output, "Enxame:\n");
	for (i = 0; i < param.quant_particulas; i++){
		for (j = 0; j < quant_func_ob; j++){

			if (param.funcoes_obj_pareto[j] == '1'){

				fprintf(output, "%f\t", enxame[i].posicao.func_ob[j]);
			}
		}
		fprintf(output, "\n");
	}
	fprintf(output, "\n");

	fprintf(output, "Soluções:\n");
	for (i = 0; i < solucao.quant_sol_pareto; i++){
		for (j = 0; j < quant_func_ob; j++){

			if (param.funcoes_obj_pareto[j] == '1'){

				fprintf(output, "%f\t", solucao.solucoes[i].func_ob[j]);
			}
		}
		fprintf(output, "\n");
	}
	fprintf(output, "\n");
}

void insereSolucoesEmArquivo(regiao_pareto solucao, parametros param, FILE* output){

	int i, j;

	fprintf(output, "Fim:\n\n");
	fprintf(output, "Soluções:\n");
	for (i = 0; i < solucao.quant_sol_pareto; i++){
		for (j = 0; j < quant_func_ob; j++){

			if (param.funcoes_obj_pareto[j] == '1'){

				fprintf(output, "%f\t", solucao.solucoes[i].func_ob[j]);
			}
		}
		fprintf(output, "\n");
	}
	fprintf(output, "\n");
}

void atualizaDominadores(regra* regras, int quant_regras, regiao_pareto pareto){

	zeraQuantDominadoresRegras(regras, quant_regras);
	int i, j;

	for (i = 0; i < quant_regras - 1; i++){

		for (j = i + 1; j < quant_regras; j++){

			if (dominaPorPareto(regras[i], regras[j], pareto) == -1){
				regras[j].quant_dominadores++;
			}
			else if (dominaPorPareto(regras[i], regras[j], pareto) == 1){
				regras[i].quant_dominadores++;
			}
		}
	}
}

void insereDominadasPorOrdemDeDominadores(parametros param, regra* regras, int quant_regras, regiao_pareto* pareto){

	atualizaDominadores(regras, quant_regras, *pareto);
	atualizaCrowdingDistances(*pareto, regras, quant_regras);
	ordenaRegrasMenoresCrowdDistances(regras, quant_regras);

	int cont = 0, i = 0;
	while ((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){

		if (regras[i].quant_dominadores == 0 && (regras[i].crowding_distance != INFINITO || cont != 1)){
			inserePareto(pareto, regras[i]);

			if (regras[i].crowding_distance == INFINITO)
				cont = 1;
		}
		i++;
	}

	ordenaRegrasMenosDominadas(regras, quant_regras);

	i = 0;
	while ((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){

		if (regras[i].quant_dominadores != 0){
			inserePareto(pareto, regras[i]);
		}
		i++;
	}
}

void insereDominadasPorOrdemDeCrowdDistance(parametros param, regra* regras, int quant_regras, regiao_pareto* pareto){

	atualizaDominadores(regras, quant_regras, *pareto);
	atualizaCrowdingDistances(*pareto, regras, quant_regras);
	ordenaRegrasMenoresCrowdDistances(regras, quant_regras);

	int i = 0;
	while ((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){

		if (regras[i].quant_dominadores == 0){
			inserePareto(pareto, regras[i]);
		}
		i++;
	}

	i = 0;

	while ((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){

		if (regras[i].quant_dominadores != 0){
			inserePareto(pareto, regras[i]);
		}
		i++;
	}
}

void removeDominadas(regiao_pareto* pareto){

	int i;
	for (i = 0; i < (*pareto).quant_sol_pareto; i++){

		if ((*pareto).solucoes[i].quant_dominadores != 0){

			(*pareto).solucoes[i].nula = 1;
		}
	}

	apagaSolucoesNulasPareto(pareto);
}

void insereNaoDominadasPorOrdemDeCrowdDistance(parametros param, regra* regras, int quant_regras, regiao_pareto* pareto){

	atualizaDominadores(regras, quant_regras, *pareto);
	atualizaCrowdingDistances(*pareto, regras, quant_regras);
	ordenaRegrasMenoresCrowdDistances(regras, quant_regras);

	int i = 0;
	while ((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){

		if (regras[i].quant_dominadores == 0){
			inserePareto(pareto, regras[i]);
		}
		i++;
	}
}

//INÍCIO DE FUNÇÕES PARALELAS//

//inicializa seeds usadas para gerar valores aleatórios nos kernels CUDA
__global__ void setup_kernel(curandState * state, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

/* gera valor intiero aleatório*/
__device__ int device_getRandom(curandState *globalState){
	return (int)truncf(curand_uniform(globalState) * 100000);
}

/* Zera todas as posições da matriz de contingência de uma regra */
__device__ void device_zeraMatrizCont(regra* r){

	int i;

	for (i = 0; i < quant_mat_cont; i++){

		(*r).mat_cont[i] = 0;
	}
}

__device__ void device_preencheMatrizCont(regra* r, exemplo* exemplos, int quant_exemp, int quant_atrib){

	device_zeraMatrizCont(r);
	int i, j, b, h;

	for (i = 0; i < quant_exemp; i++){

		b = 1;
		h = 1;

		for (j = 0; j < quant_atrib - 1; j++){

			if (((*(exemplos + i)).campos[j] != (*r).valores[j]) && ((*r).valores[j] != -1)){
				b = 0;
			}
		}

		if ((*(exemplos + i)).campos[quant_atrib - 1] != (*r).valores[quant_atrib - 1]){
			h = 0;
		}

		if (b == 1)
			(*r).mat_cont[B]++;
		else
			(*r).mat_cont[_B]++;

		if (h == 1)
			(*r).mat_cont[H]++;
		else
			(*r).mat_cont[_H]++;

		if (b == 1 && h == 1)
			(*r).mat_cont[BH]++;

		else if (b == 1 && h == 0)
			(*r).mat_cont[B_H]++;

		else if (b == 0 && h == 1)
			(*r).mat_cont[_BH]++;

		else if (b == 0 && h == 0)
			(*r).mat_cont[_B_H]++;
	}
}

__device__ void device_calculaFuncoesObj(regra* r, int quant_atrib, atributo* atrib, int quant_exemp){
	int quant_classes = (*(atrib + quant_atrib - 1)).quant_real;

	if ((*r).mat_cont[B] != 0)
		(*r).func_ob[ACC] = (double)((*r).mat_cont[BH]) / (*r).mat_cont[B];
	else
		(*r).func_ob[ACC] = -1;
	if ((*r).mat_cont[B] != 0)
		(*r).func_ob[ERR] = (double)((*r).mat_cont[B_H]) / (*r).mat_cont[B];
	else
		(*r).func_ob[ERR] = -1;
	if ((*r).mat_cont[_B] != 0)
		(*r).func_ob[NEGREL] = (double)((*r).mat_cont[_B_H]) / (*r).mat_cont[_B];
	else
		(*r).func_ob[NEGREL] = -1;
	if (((*r).mat_cont[B] + quant_classes) != 0)
		(*r).func_ob[ACCLP] = (double)((*r).mat_cont[BH] + 1) / ((*r).mat_cont[B] + quant_classes);
	else
		(*r).func_ob[ACCLP] = -1;
	if ((*r).mat_cont[H] != 0)
		(*r).func_ob[SENS] = (double)((*r).mat_cont[BH]) / (*r).mat_cont[H];
	else
		(*r).func_ob[SENS] = -1;
	if ((*r).mat_cont[_H] != 0)
		(*r).func_ob[SPEC] = (double)((*r).mat_cont[_B_H]) / (*r).mat_cont[_H];
	else
		(*r).func_ob[SPEC] = -1;
	if (quant_exemp != 0)
		(*r).func_ob[COV] = (double)((*r).mat_cont[B]) / quant_exemp;
	else
		(*r).func_ob[COV] = -1;
	if (quant_exemp != 0)
		(*r).func_ob[SUP] = (double)((*r).mat_cont[BH]) / quant_exemp;
	else
		(*r).func_ob[SUP] = -1;
}

__device__ double** device_inicializaVelocidadeParticula(atributo* atrib, int quant_atrib, curandState* globalState){

	double** vel = (double**)malloc(quant_atrib * sizeof(double*));
	if (vel == NULL){
		printf("\nErro de alcocacao! 'device_inicializaVelocidadeParticula'");
	}

	int i, j;
	for (i = 0; i < quant_atrib; i++){

		vel[i] = (double*)malloc((atrib[i].quant_real + 1) * sizeof(double));
		if (vel[i] == NULL){
			printf("\nErro de alcocacao! 'device_inicializaVelocidadeParticula'");
		}
		for (j = 0; j < atrib[i].quant_real + 1; j++){

			vel[i][j] = (device_getRandom(globalState) % 101) / 100.0;
		}
	}

	return vel;
}

__device__ regra device_inicializaRegra(){
	regra r;
	int i;

	for (i = 0; i < quant_mat_cont; i++)																						// inicializa com 0's a matriz de contigência;
		r.mat_cont[i] = 0;

	r.nula = 1;																												// regra inicialmete é nula;
	r.crowding_distance = 0;
	r.quant_dominadores = 0;

	return r;
}

__device__ regra device_geraRegra(parametros param, atributo* atrib, int quant_atrib, int classe, curandState* globalState){

	regra r = device_inicializaRegra();
	int i, limiar;

	//srand( (unsigned) time(NULL) );
	for (i = 0; i < quant_atrib - 1; i++){

		if (param.metodo_gerar_regras == 1){

			limiar = device_getRandom(globalState) % 101;
			if (param.prob_valor_vazio >= limiar)
				r.valores[i] = -1;
			else
				r.valores[i] = device_getRandom(globalState) % ((*(atrib + i)).quant_real);
		}
		else if (param.metodo_gerar_regras == 0)
			r.valores[i] = device_getRandom(globalState) % ((*(atrib + i)).quant_real + 1) - 1;
	}

	switch (classe){

	case 0:
		r.valores[quant_atrib - 1] = 0;
		break;

	case 1:
		r.valores[quant_atrib - 1] = 1;
		break;

	default:
		r.valores[quant_atrib - 1] = device_getRandom(globalState) % ((*(atrib + i)).quant_real);												// valor aleatório para a classe;
	}

	r.nula = 0;																												// regra deixa de ser nula;

	return r;
}

__device__ regra* device_geraRegras(parametros param, int quant_regras, atributo* atrib, int quant_atrib, int classe, exemplo* exemplos, int quant_exemp, curandState* globalState){

	int i;
	regra* regras = (regra*)malloc(quant_regras * sizeof(regra));
	if (regras == NULL){
		printf("\nErro de alcocacao! 'device_geraRegras'");
	}

	for (i = 0; i < quant_regras; i++){

		*(regras + i) = device_geraRegra(param, atrib, quant_atrib, classe, globalState);
		device_preencheMatrizCont((regras + i), exemplos, quant_exemp, quant_atrib);
		device_calculaFuncoesObj((regras + i), quant_atrib, atrib, quant_exemp);
		//imprimeRegra(*(regras + i), quant_atrib);
		//imprimeMatrizCont(*(regras + i));
		//imprimeFuncoesObj(*(regras + i));
	}

	return regras;
}

/*Dado dois vetores de regras v1 e v2 e seus respectivos tamanhos t1 e t2,
retorna um novo vetor v de tamanho t1 + t2 contendo as regras de v1 e v2*/
__device__ regra* device_uneRegras(regra* regras1, int quant_regras1, regra* regras2, int quant_regras2){

	regra* regras = (regra*)malloc((quant_regras1 + quant_regras2) * sizeof(regra));
	if (regras == NULL){

		printf("\nErro de alcocacao! 'device_uneRegras'");
	}

	int i;
	for (i = 0; i < quant_regras1; i++){

		if (regras1[i].nula != 1)
			regras[i] = regras1[i];
	}
	for (i = 0; i < quant_regras2; i++){

		if (regras2[i].nula != 1)
			regras[i + quant_regras1] = regras2[i];
	}

	return regras;
}

__device__ int device_verificaIgualdadeRegras(regra r1, regra r2, int quant_atrib){

	int iguais = 1;
	int i;

	for (i = 0; i < quant_atrib; i++){

		if (r1.valores[i] != r2.valores[i])
			iguais = 0;
	}

	return iguais;
}

__device__ regra* device_apagaRegrasIguais(regra* r, int* quant_regras, int quant_atrib){

	int i, j;

	for (i = 0; i < *quant_regras; i++){
		for (j = 0; j < *quant_regras; j++){

			if ((i != j) && (r[i].nula == 0) && (r[j].nula == 0) && (device_verificaIgualdadeRegras(r[i], r[j], quant_atrib) == 1)){

				r[i].nula = 1;
			}
		}
	}

	int cont = 0;

	for (i = 0; i < *quant_regras; i++){

		if (r[i].nula == 0)
			cont++;
	}
	//printf("\n%d %d", *quant_regras, cont);
	regra* regras = (regra*)malloc(cont * sizeof(regra));
	if (regras == NULL){

		printf("\nErro de alcocacao! 'device_apagaRegrasIguais'");
	}

	cont = 0;

	for (i = 0; i < *quant_regras; i++){

		if (r[i].nula == 0){

			regras[cont] = r[i];
			cont++;
		}
	}

	*quant_regras = cont;

	return regras;
}

/*Retorna -1 se r1 domina r2, 0 se nenhuma das duas regras domina a outra, 1 se r2 domina r1*/
__device__ int device_dominaPorPareto(regra r1, regra r2, regiao_pareto pareto){

	int i;
	int saida = 0;

	for (i = 0; i < quant_func_ob; i++){
		if (pareto.func_obj[i] == 1){
			if (r1.func_ob[i] > r2.func_ob[i]){

				if (saida == 0 || saida == -1)
					saida = -1;
				else
					return 0;
			}
			if (r1.func_ob[i] < r2.func_ob[i]){

				if (saida == 0 || saida == 1)
					saida = 1;
				else
					return 0;
			}
		}
	}

	return saida;
}

__device__ int device_contaRegrasNaoNulasPareto(regiao_pareto pareto){

	int cont = 0;
	int i;

	for (i = 0; i < pareto.quant_sol_pareto; i++){

		if (pareto.solucoes[i].nula == 0)
			cont++;
	}

	return cont;
}

__device__ void device_apagaSolucoesNulasPareto(regiao_pareto* pareto){

	int quant_sol_nao_nulas = device_contaRegrasNaoNulasPareto(*pareto);
	regra* solucoes = (regra*)malloc(quant_sol_nao_nulas * sizeof(regra));
	if (solucoes == NULL){

		printf("\nErro de alcocacao! 'device_apagaSolucoesNulasPareto'");
	}

	int cont = 0;
	int i;

	for (i = 0; i < (*pareto).quant_sol_pareto; i++){

		if ((*pareto).solucoes[i].nula == 0){

			solucoes[cont] = (*pareto).solucoes[i];
			cont++;
		}
	}

	(*pareto).solucoes = solucoes;
	(*pareto).quant_sol_pareto = quant_sol_nao_nulas;
}

__device__ void device_removeDominadas(regiao_pareto* pareto){

	int i;
	for (i = 0; i < (*pareto).quant_sol_pareto; i++){

		if ((*pareto).solucoes[i].quant_dominadores != 0){

			(*pareto).solucoes[i].nula = 1;
		}
	}

	device_apagaSolucoesNulasPareto(pareto);
}

__device__ particula device_criaParticula(parametros param, atributo* atrib, int quant_atrib, curandState* globalState){

	particula p;
	p.posicao = device_geraRegra(param, atrib, quant_atrib, param.classe, globalState);
	p.lBest = p.posicao;
	p.gBest = p.posicao;
	p.gBest.nula = 1;
	p.velocidade = device_inicializaVelocidadeParticula(atrib, quant_atrib, globalState);

	return p;
}

__device__ regra* device_enxameParaRegras(parametros param, particula* enxame){

	regra* regras = (regra*)malloc(param.quant_particulas * sizeof(regra));
	if (regras == NULL){

		printf("\nErro de alcocacao! 'device_enxameParaRegras'");
	}

	int i;
	for (i = 0; i < param.quant_particulas; i++){

		regras[i] = enxame[i].posicao;
	}

	return regras;
}

__device__ void device_setGBest(parametros param, regiao_pareto pareto, particula* enxame, curandState* globalState){

	int i;
	regra r1, r2;

	for (i = 0; i < param.quant_particulas; i++){

		r1 = pareto.solucoes[device_getRandom(globalState) % pareto.quant_sol_pareto];
		r2 = pareto.solucoes[device_getRandom(globalState) % pareto.quant_sol_pareto];

		if (r1.crowding_distance > r2.crowding_distance)
			enxame[i].gBest = r1;
		else
			enxame[i].gBest = r2;
	}
}

__device__ double device_getOmega(){

	return MAX_OMEGA;
}

__device__ double device_getPhi(curandState* globalState){

	double phi;

	do{
		phi = (device_getRandom(globalState) % 101) / 100.0;
	} while (phi > MAX_OMEGA);

	return phi;
}

__device__ double device_getConst(){

	return 2;
}

__device__ int device_posicaoMenosPosicao(int p1, int p2){
	if (p1 != p2)
		return p1;
	else
		return -2;	// -2 representa um valor fora do domínio dos atributo da regra;
}

__device__ double device_coefVezesPosicao(double c, int posicao, curandState* globalState){

	double phi = device_getPhi(globalState);
	if (posicao == -2)
		return 0;
	else if (c*phi < 1)
		return c*phi;
	else
		return 1;
}

__device__ void device_coefVezesVelocidade(double* velocidade, int quant_vel){

	double coef = device_getOmega();
	int i;

	for (i = 0; i < quant_vel; i++){

		if (coef*velocidade[i] < 1)
			velocidade[i] = coef*velocidade[i];
		else
			velocidade[i] = 1;
	}
}

/*Dada a velocidade calculada de um valor de um atributo, verifica
se esse valor é maior que o correspondente dentro do vetor de velocidades*/
__device__ void device_somaVelocidades(double* velocidades, double vel, int atributo){
	if (velocidades[atributo + 1] < vel)
		velocidades[atributo + 1] = vel;
}

__device__ void device_setVelocidadeParticula(particula* p, atributo* atrib, int quant_atrib, parametros param, curandState* globalState){

	double c1 = device_getConst();
	double c2 = device_getConst();

	int i;
	for (i = 0; i < quant_atrib; i++){

		device_coefVezesVelocidade((*p).velocidade[i], atrib[i].quant_real + 1);
		device_somaVelocidades((*p).velocidade[i], device_coefVezesPosicao(c1, device_posicaoMenosPosicao((*p).lBest.valores[i], (*p).posicao.valores[i]), globalState), (*p).lBest.valores[i]);
		device_somaVelocidades((*p).velocidade[i], device_coefVezesPosicao(c2, device_posicaoMenosPosicao((*p).gBest.valores[i], (*p).posicao.valores[i]), globalState), (*p).gBest.valores[i]);
	}
}

__device__ double device_getMaiorVelocidade(double* velocidades, int quant_vel){

	double vel = 0;
	int i;

	for (i = 0; i < quant_vel; i++){

		if (velocidades[i] > vel)
			vel = velocidades[i];
	}

	return vel;
}

__device__ double device_getRandFloat(double limite, curandState* globalState){

	double r;

	do{
		r = (device_getRandom(globalState) % 101) / 100.0;
	} while (r > limite);

	return r;
}

__device__ int device_roleta(double* velocidade, atributo atrib, curandState* globalState){

	double limite = device_getMaiorVelocidade(velocidade, atrib.quant_real + 1);
	double alfa = device_getRandFloat(limite, globalState);
	int posicao;

	do{
		posicao = device_getRandom(globalState) % (atrib.quant_real + 1) - 1;
	} while (velocidade[posicao + 1] < alfa);

	return posicao;
}

__device__ void device_setPosicaoParticula(particula* p, atributo* atrib, int quant_atrib, curandState* globalState){

	int i;
	for (i = 0; i < quant_atrib; i++){
		(*p).posicao.valores[i] = device_roleta((*p).velocidade[i], atrib[i], globalState);
	}
}

__device__ regiao_pareto device_inicializaPareto(parametros param){

	regiao_pareto pareto;

	pareto.solucoes = (regra*)malloc(1 * sizeof(regra));
	if (pareto.solucoes == NULL){

		printf("\nErro de alcocacao! 'device_inicializaPareto'");
	}

	pareto.solucoes[0].nula = 1;
	pareto.quant_sol_pareto = 1;

	int i;
	for (i = 0; i < quant_func_ob; i++){
		pareto.func_obj[i] = (param.funcoes_obj_pareto[i]) - 48;
	}

	return pareto;
}

__device__ void device_inserePareto(regiao_pareto* pareto, regra r){

	int i, inserido = 0;

	for (i = 0; i < (*pareto).quant_sol_pareto; i++){

		if ((*pareto).solucoes[i].nula == 1){

			(*pareto).solucoes[i] = r;
			(*pareto).solucoes[i].nula = 0;
			inserido = 1;
			break;
		}
	}

	if (inserido == 0){

		//cria cópia da array de regras
		int quantidade_antiga = (*pareto).quant_sol_pareto;
		regra* copia_solucoes = (regra*)malloc(quantidade_antiga*sizeof(regra));
		for (i = 0; i < quantidade_antiga; i++){
			copia_solucoes[i] = (*pareto).solucoes[i];
		}

		//realoca array de regras com 1 unidade de memória a mais e carrega valores da cópia
		free((*pareto).solucoes);
		(*pareto).solucoes = (regra*)malloc((quantidade_antiga + 1)*sizeof(regra));
		for (i = 0; i < quantidade_antiga; i++){
			(*pareto).solucoes[i] = copia_solucoes[i];
		}

		//insere nova regra na nova posição alocada
		(*pareto).quant_sol_pareto++;
		(*pareto).solucoes[quantidade_antiga] = r;
		(*pareto).solucoes[quantidade_antiga].nula = 0;
	}
}

__device__ void device_zeraQuantDominadoresRegras(regra* regras, int quant_regras){

	int i;
	for (i = 0; i < quant_regras; i++){
		regras[i].quant_dominadores = 0;
	}
}

__device__ int device_comparaDominadores(const void *a, const void *b){
	regra *x = (regra *)a;
	regra *y = (regra *)b;

	if ((*x).quant_dominadores >(*y).quant_dominadores)
		return 1;
	if ((*x).quant_dominadores < (*y).quant_dominadores)
		return -1;
	return 0;
}

__device__ void device_swap(void *e1, void *e2, int size) {
	void* sp = (void*)malloc(size);
	memcpy(sp, e1, size);
	memcpy(e1, e2, size);
	memcpy(e2, sp, size);
	free(sp);
}

__device__ void device_qsort(void* base, size_t n, size_t size, int(*cmp)(const void*, const void*)) {
	int i, j;

	if (n <= 1) {
		return;
	}
	i = 0; j = n;
	while (1) {
		do {
			++i;
		} while (cmp((char*)base + size*i, base) < 0 && i < n);
		do {
			--j;
		} while (cmp((char*)base + size*j, base) > 0);   //no need to check j >= 0 because it will fail test when it reaches base
		if (i > j) break;
		device_swap((char*)base + size*i, (char*)base + size*j, size);
	}
	//swap the base element with the j element
	if (j != 0) {
		device_swap(base, (char*)base + size*j, size);
	}
	if (j > 0) {
		device_qsort(base, j, size, cmp);
	}
	device_qsort((char*)base + size*(j + 1), n - 1 - j, size, cmp);
}

__device__ void device_ordenaRegrasMenosDominadas(regra* regras, int quant_regras){
	device_qsort(regras, quant_regras, sizeof(regra), device_comparaDominadores);
}

__device__ int device_comparaCrowdDistances(const void *a, const void *b){
	regra *x = (regra *)a;
	regra *y = (regra *)b;

	if ((*x).crowding_distance > (*y).crowding_distance)
		return -1;
	if ((*x).crowding_distance < (*y).crowding_distance)
		return 1;
	return 0;
}

__device__ void device_ordenaRegrasMenoresCrowdDistances(regra* regras, int quant_regras){
	device_qsort(regras, quant_regras, sizeof(regra), device_comparaCrowdDistances);
}

__device__ void device_atualizaCrowdingDistance(regiao_pareto pareto, int objetivo, regra* regras, int quant_regras){

	int i, j;
	/* imprimeDominioParetoComObjetivos(*pareto, quant_atrib, param);

	printf("\nAntigas Crowding Distances:\n");
	for(j = 0; j < (*pareto).quant_sol_pareto; j++){

	printf("%f ", (*pareto).solucoes[j].crowding_distance);
	} */

	switch (objetivo){
	case ACC:
		break;
	case ERR:
		break;
	case NEGREL:
		break;
	case ACCLP:
		break;
	case SENS:
		device_qsort(regras, quant_regras, sizeof(regra), comparaSENS);
		break;
	case SPEC:
		device_qsort(regras, quant_regras, sizeof(regra), comparaSPEC);
		break;
	case COV:
		break;
	case SUP:
		break;
	}

	for (i = 0; i < quant_regras; i++){
		if (regras[i].func_ob[objetivo] == regras[0].func_ob[objetivo] || regras[i].func_ob[objetivo] == regras[quant_regras - 1].func_ob[objetivo])
			regras[i].crowding_distance = INFINITO;
	}

	//regras[0].crowding_distance = INFINITO;
	for (i = 1; i < quant_regras - 1; i++){
		if (regras[i].crowding_distance != INFINITO)
			regras[i].crowding_distance += regras[i + 1].func_ob[objetivo] - regras[i - 1].func_ob[objetivo];
	}
	//regras[quant_regras - 1].crowding_distance = INFINITO;

	/* printf("\nNovas Crowding Distances:\n");
	for(j = 0; j < (*pareto).quant_sol_pareto; j++){

	printf("%f ", (*pareto).solucoes[j].crowding_distance);
	} */
}

__device__ void device_atualizaCrowdingDistances(regiao_pareto pareto, regra* regras, int quant_regras){

	int i;
	for (i = 0; i < quant_func_ob; i++){

		if (pareto.func_obj[i] == 1)
			device_atualizaCrowdingDistance(pareto, i, regras, quant_regras);
	}

}

__device__ void device_atualizaDominadores(regra* regras, int quant_regras, regiao_pareto pareto){

	device_zeraQuantDominadoresRegras(regras, quant_regras);
	int i, j;

	for (i = 0; i < quant_regras - 1; i++){

		for (j = i + 1; j < quant_regras; j++){

			if (device_dominaPorPareto(regras[i], regras[j], pareto) == -1){
				regras[j].quant_dominadores++;
			}
			else if (device_dominaPorPareto(regras[i], regras[j], pareto) == 1){
				regras[i].quant_dominadores++;
			}
		}
	}
}

__device__ void device_insereDominadasPorOrdemDeDominadores(parametros param, regra* regras, int quant_regras, regiao_pareto* pareto){

	device_atualizaDominadores(regras, quant_regras, *pareto);
	device_atualizaCrowdingDistances(*pareto, regras, quant_regras);
	device_ordenaRegrasMenoresCrowdDistances(regras, quant_regras);

	int cont = 0, i = 0;
	while ((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){

		if (regras[i].quant_dominadores == 0 && (regras[i].crowding_distance != INFINITO || cont != 1)){
			device_inserePareto(pareto, regras[i]);

			if (regras[i].crowding_distance == INFINITO)
				cont = 1;
		}
		i++;
	}

	device_ordenaRegrasMenosDominadas(regras, quant_regras);

	i = 0;
	while ((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){

		if (regras[i].quant_dominadores != 0){
			device_inserePareto(pareto, regras[i]);
		}
		i++;
	}
}

__device__ void device_insereDominadasPorOrdemDeCrowdDistance(parametros param, regra* regras, int quant_regras, regiao_pareto* pareto){

	device_atualizaDominadores(regras, quant_regras, *pareto);
	device_atualizaCrowdingDistances(*pareto, regras, quant_regras);
	device_ordenaRegrasMenoresCrowdDistances(regras, quant_regras);

	int i = 0;
	while ((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){

		if (regras[i].quant_dominadores == 0){
			device_inserePareto(pareto, regras[i]);
		}
		i++;
	}

	i = 0;

	while ((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){

		if (regras[i].quant_dominadores != 0){
			device_inserePareto(pareto, regras[i]);
		}
		i++;
	}
}

__device__ void device_insereNaoDominadasPorOrdemDeCrowdDistance(parametros param, regra* regras, int quant_regras, regiao_pareto* pareto){

	device_atualizaDominadores(regras, quant_regras, *pareto);
	device_atualizaCrowdingDistances(*pareto, regras, quant_regras);
	device_ordenaRegrasMenoresCrowdDistances(regras, quant_regras);

	int i = 0;
	while ((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){

		if (regras[i].quant_dominadores == 0){
			device_inserePareto(pareto, regras[i]);
		}
		i++;
	}
}

__global__ void criaEnxames(parametros param, exemplo* exemplos, int quant_exemp, atributo* atrib, int quant_atrib, particula* enxame, curandState* globalState){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > param.quant_particulas * param.quant_enxames)
		return;

	enxame[i] = device_criaParticula(param, atrib, quant_atrib, &globalState[i]);
}

__global__ void SPSOMultiEnxame(parametros* param, exemplo* exemplos, int quant_exemp, atributo* atrib, int quant_atrib, FILE* output, particula* enxames, regra* posicoesFinais, curandState* globalState){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id > (*param).quant_particulas * (*param).quant_enxames)
		return;

	extern __shared__ int mem[];	//memória compartilhada que armazenará as variáveis de todas as arrays dinâmicas utilizadas
	particula *enxame_shared = (particula*)&mem[0];	//lista de particulas correspondente ao enxame de cada bloco

	//CARREGA PARTÍCULA, CALCULA FUNÇÕES OBJETIVO E PREENCHE ENXAME DO RESPECTIVO BLOCO

	particula p = enxames[id];
	device_preencheMatrizCont(&(p.posicao), exemplos, quant_exemp, quant_atrib);
	device_calculaFuncoesObj(&(p.posicao), quant_atrib, atrib, quant_exemp);
	enxame_shared[threadIdx.x] = p;

	__syncthreads();

	//CALCULA MELHORES PARTÍCULAS DO ENXAME

	regra* regras_enxame;
	regiao_pareto pareto;

	if (threadIdx.x == 0)	//caso seja a primeira thread do bloco (a primeira partícula do enxame)
	{
		regras_enxame = device_enxameParaRegras((*param), enxame_shared);
		pareto = device_inicializaPareto((*param));

		if ((*param).metodo_dopagem_solucao == 0)	// ----->> insere dominadas na solução (insereDominadasPorOrdemDeCrowdDistance ou insereDominadasPorOrdemDeDominadores())
			device_insereDominadasPorOrdemDeDominadores((*param), regras_enxame, (*param).quant_particulas, &pareto);
		else if ((*param).metodo_dopagem_solucao == 1)
			device_insereDominadasPorOrdemDeCrowdDistance((*param), regras_enxame, (*param).quant_particulas, &pareto);
		else
			device_insereNaoDominadasPorOrdemDeCrowdDistance((*param), regras_enxame, (*param).quant_particulas, &pareto);
		device_setGBest((*param), pareto, enxame_shared, globalState);
	}

	__syncthreads();

	//INICIA LAÇO EVOLUTIVO

	regra* total;

	int i, quant_regras;
	for (i = 0; i < (*param).bl_interacoes; i++){
		if ((*param).classe == -1){

			device_setVelocidadeParticula(&enxame_shared[threadIdx.x], atrib, quant_atrib, (*param), globalState);
			device_setPosicaoParticula(&enxame_shared[threadIdx.x], atrib, quant_atrib, globalState);
		}
		else{
			device_setVelocidadeParticula(&enxame_shared[threadIdx.x], atrib, quant_atrib - 1, (*param), globalState);
			device_setPosicaoParticula(&enxame_shared[threadIdx.x], atrib, quant_atrib - 1, globalState);
		}

		device_preencheMatrizCont(&(enxame_shared[threadIdx.x].posicao), exemplos, quant_exemp, quant_atrib);
		device_calculaFuncoesObj(&(enxame_shared[threadIdx.x].posicao), quant_atrib, atrib, quant_exemp);

		if (device_dominaPorPareto(enxame_shared[threadIdx.x].posicao, enxame_shared[threadIdx.x].lBest, pareto) == -1)
			enxame_shared[threadIdx.x].lBest = enxame_shared[threadIdx.x].posicao;

		__syncthreads();

		//ATUALIZA DOMÍNIO DE PARETO DO ENXAME

		if (threadIdx.x == 0)	//caso seja a primeira thread do bloco (a primeira partícula do enxame)
		{
			regras_enxame = device_enxameParaRegras((*param), enxame_shared);
			total = device_uneRegras(regras_enxame, (*param).quant_particulas, pareto.solucoes, pareto.quant_sol_pareto);
			quant_regras = (*param).quant_particulas + pareto.quant_sol_pareto;
			total = device_apagaRegrasIguais(total, &quant_regras, quant_atrib);
			pareto = device_inicializaPareto((*param));

			if ((*param).metodo_dopagem_solucao == 0)																		// ----->> insere dominadas na solução (insereDominadasPorOrdemDeCrowdDistance ou insereDominadasPorOrdemDeDominadores())
				device_insereDominadasPorOrdemDeDominadores((*param), total, quant_regras, &pareto);
			else if ((*param).metodo_dopagem_solucao == 1)
				device_insereDominadasPorOrdemDeCrowdDistance((*param), total, quant_regras, &pareto);
			else
				device_insereNaoDominadasPorOrdemDeCrowdDistance((*param), total, quant_regras, &pareto);

			device_setGBest((*param), pareto, enxame_shared, globalState);
			//insereObjEmArquivo(pareto, enxame, param, output, i + 1);
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)	//caso seja a primeira thread do bloco (a primeira partícula do enxame)
	{
		device_removeDominadas(&pareto);
	}

	__syncthreads();

	//INSERE POSIÇÃO FINAL DE CADA PARTÍCULA NO VETOR DE POSIÇÕES FINAIS
	posicoesFinais[id] = enxame_shared[threadIdx.x].posicao;
}

//FIM DE FUNÇÕES PARALELAS//

//////////////////////////////////////// FIM S-PSO ////////////////////////////////////////

void criaArquivoObjSolucao(regiao_pareto solucao, int indice_obj, int indice_arq){

	char nome_aqr[22];
	sprintf(nome_aqr, "objetivo(%d).txt", indice_arq);

	FILE* output = fopen(nome_aqr, "w");

	if (output == NULL){

		printf("Nao foi possivel criar arquivo!\n");
		printf("\n%s", nome_aqr);
		printf("\nlenght = %d", strlen(nome_aqr));
		system("pause");
		exit(1);
	}
	else{
		fprintf(output, "%s\n", nome_aqr);

		int i;
		for (i = 0; i < solucao.quant_sol_pareto; i++){

			fprintf(output, "%f\n", solucao.solucoes[i].func_ob[indice_obj]);
		}

		fclose(output);
	}
}

void criaArquivosObjSolucao(regiao_pareto solucao, parametros param){

	int cont = 1;
	int i, j;

	for (i = 0; i < quant_func_ob; i++){

		if (param.funcoes_obj_pareto[i] == '1'){

			criaArquivoObjSolucao(solucao, i, cont);
			cont++;
		}
	}
}

void imprimeSolucao(regiao_pareto solucao, atributo* atrib, int quant_atrib){

	printf("\n#SOLUÇÔES SPSO:\n");

	int i;
	for (i = 0; i < solucao.quant_sol_pareto; i++){

		imprimeNomesRegra(solucao.solucoes[i], atrib, quant_atrib);
	}
}

//////////////////////////////////////// INÍCIO DE CLASSIFICAÇÃO ////////////////////////////////////////

classificador inicializaClassificador(regra* regras, int quant_regras){
	classificador c;
	c.quant_regras = quant_regras;
	c.regras = regras;

	int i;
	for (i = 0; i < quant_mat_conf; i++)
		c.mat_conf[i] = 0;

	return c;
}

int comparaEspecificidade(const void *a, const void *b){
	regra *x = (regra *)a;
	regra *y = (regra *)b;

	if ((*x).func_ob[SPEC] >(*y).func_ob[SPEC])
		return -1;
	if ((*x).func_ob[SPEC] < (*y).func_ob[SPEC])
		return 1;
	return 0;
}

void ordenaRegrasMaiorEspecificidade(regra* regras, int quant_regras){
	qsort(regras, quant_regras, sizeof(regra), comparaEspecificidade);
}

/*Retorna 0 se a regra não cobre o exemplo ou 1 caso contrário.*/
int cobreExemplo(regra r, exemplo e, int quant_atrib){
	int i;
	for (i = 0; i < quant_atrib - 1; i++){
		if ((e.campos[i] != r.valores[i]) && (r.valores[i] != -1)){
			return 0;
		}
	}
	return 1;
}

regra* buscaRegrasVotantes(regra* regrasClass, int quant_regras_class, exemplo e, int quant_atrib, int quant_max_regras_vot){
	regra *regrasVot = (regra*)calloc(quant_max_regras_vot, sizeof(regra));

	//inicializa regras como nulas
	regra rNula;
	rNula.nula = 1;

	int i;
	for (i = 0; i < quant_max_regras_vot; i++)
		regrasVot[i] = rNula;

	int positivas = 0;
	int negativas = 0;

	//busca votantes da classe do exemplo
	for (i = 0; i < quant_regras_class; i++){
		regra r = regrasClass[i];
		if (cobreExemplo(r, e, quant_atrib)){
			regrasVot[positivas + negativas] = r;
			if (positivas < quant_max_regras_vot / 2 && r.valores[quant_atrib - 1] == 0)	positivas++;
			else if (negativas < quant_max_regras_vot / 2 && r.valores[quant_atrib - 1] == 1) negativas++;

			if (positivas >= quant_max_regras_vot / 2 && negativas >= quant_max_regras_vot / 2)
				break;
		}
	}
	return regrasVot;
}

int calculaVotacao(regra* votantes, int quant_votantes, int func_ob, int quant_atrib){
	double votos = 0;
	int i;
	for (i = 0; i < quant_votantes; i++){
		regra r = votantes[i];
		if (r.nula == 0){
			if (r.valores[quant_atrib - 1] == 0)	votos += r.func_ob[func_ob];	//voto para classe positiva
			else votos -= r.func_ob[func_ob];	//voto para classe negativa
		}
	}

	if (votos > 0) return 1;	//caso em que votaram na classe positiva
	else if (votos < 0) return 0;	//caso em que votaram na classe negativa
	else return rand() % 2;	//caso em que não houve votação (vota aleatoriamente)
}

void classificaExemplos(classificador* c, exemplo* exemplos, int quant_exemp, int quant_atrib){

	int quant_regras_vot = 2 * 2;	// 2 vezes número de classes K (K positivas e K negativas)

	int i;
	for (i = 0; i < quant_exemp; i++){
		regra* votantes = buscaRegrasVotantes((*c).regras, (*c).quant_regras, exemplos[i], quant_atrib, quant_regras_vot);
		int classe = calculaVotacao(votantes, quant_regras_vot, SPEC, quant_atrib);
		int classeReal = exemplos[i].campos[quant_atrib - 1];

		if (classe == 0 && classeReal == 0) (*c).mat_conf[TP]++;
		else if (classe == 0 && classeReal == 1) (*c).mat_conf[FP]++;
		else if (classe == 1 && classeReal == 1) (*c).mat_conf[TN]++;
		else if (classe == 1 && classeReal == 0) (*c).mat_conf[FN]++;
	}
}

//////////////////////////////////////// FIM DE CLASSIFICAÇÃO ////////////////////////////////////////

int main(){

	system("cls");

	//AUMENTA LIMITES DE MEMÓRIA

	size_t size_heap, size_stack;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 200000000 * sizeof(double));
	cudaDeviceSetLimit(cudaLimitStackSize, 10 * 2048);
	cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
	cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);
	printf("Heap size found to be %d; Stack size found to be %d\n", (int)size_heap, (int)size_stack);

	//ALOCA E CARREGA VARIÁVEIS

	//carrega parâmetros
	FILE* arq_parametros = carregarArq("parametros.txt");
	parametros h_param;
	carregaParametros(arq_parametros, &h_param);
	imprimeParametros(h_param);

	parametros *d_param;	//endereço de memória no device que armazenará a cópia de h_param
	cudaMalloc(&d_param, sizeof(parametros));

	char *d_funcoes_obj_pareto;	//endereço de memória no device que armazenará a cópia da string funcoes_obj_pareto contida na struct h_param
	cudaMalloc(&d_funcoes_obj_pareto, quant_func_ob * sizeof(char));
	cudaMemcpy(d_funcoes_obj_pareto, (h_param.funcoes_obj_pareto), quant_func_ob * sizeof(char), cudaMemcpyHostToDevice);

	h_param.funcoes_obj_pareto = d_funcoes_obj_pareto;	//passa referência de string já no device e copia struct interia para a variável do device
	cudaMemcpy(d_param, &h_param, sizeof(parametros), cudaMemcpyHostToDevice);
	carregaParametros(arq_parametros, &h_param); //recupera referência às funções objetivo originais presentes no device

	//cria diretório onde serão colocados todos os arquivos de saída
	const int tamanhoDiretorioSaida = 40;
	char diretorioSaida[tamanhoDiretorioSaida];
	sprintf(diretorioSaida, "saida-%s", h_param.arquivo);
	CreateDirectory(diretorioSaida, NULL);

	//inicializa arquivo onde serão escritos os dados das matrizes de confusão referentes aos testes em cada partição.
	char nomeArquivoMatrizConf[20 + tamanhoDiretorioSaida];
	sprintf(nomeArquivoMatrizConf, "%s/matriz_confusao.txt", diretorioSaida);
	FILE* outputMatrizConf = fopen(nomeArquivoMatrizConf, "w");

	//inicializa arquivo onde serão escritos os tempos de execução de processamento.
	char nomeArquivoTemposExecucao[40 + tamanhoDiretorioSaida];
	sprintf(nomeArquivoTemposExecucao, "%s/tempos_execucao.txt", diretorioSaida);
	FILE* outputTempo = fopen(nomeArquivoTemposExecucao, "w");

	//INICIA PROCESSAMENTO POR PARTIÇÕES DE ARQUIVOS DE TREINAMENTO E DE TESTE

	for (int i = 0; i < 10; i++){

		//carrega arquivos de entrada de exemplos
		FILE* input = carregarArqDados(h_param.arquivo, i);

		//carrega arquivo que receberá as regras de saída (soluções)
		char nomeArquivoSaida[20 + tamanhoDiretorioSaida];
		sprintf(nomeArquivoSaida, "%s/saida%d.txt", diretorioSaida, i);
		FILE* output = fopen(nomeArquivoSaida, "w");

		//carrega arquito de teste do classificador final
		FILE* teste = carregarArqTeste(h_param.arquivo, i);

		//carrega atributos
		int quant_atrib = contaAtributos(input);	// quantidade de atributos dos elementos da base de dados;
		atributo *atrib_h, *atrib_d;	// ponteiros para alocação dinâmica do vetor com atributos;
		cudaMallocHost((void**)&atrib_h, sizeof(atributo) * quant_atrib);
		cudaMalloc(&atrib_d, sizeof(atributo) * quant_atrib);
		if (atrib_h == NULL || atrib_d == NULL){
			printf("\nErro de alcocacao!");
			system("pause");
		}
		processaAtributos(atrib_h, quant_atrib, input);	//preenche vetor de atributos da classe;
		//imprimeAtributos(atrib_h, quant_atrib);	//imprime todas as informações contidas no vetor de atributos;
		cudaMemcpy(atrib_d, atrib_h, sizeof(atributo) * quant_atrib, cudaMemcpyHostToDevice);

		//carrega exemplos de treino
		int quant_exemp = contaExemplos(input);																						// quantidade de exemplos da base de dados;
		exemplo *exemplos_h, *exemplos_d;																							// ponteiros para alocação dinâmica do vetor com exemplos;
		cudaMallocHost((void**)&exemplos_h, sizeof(exemplo) * quant_exemp);
		cudaMalloc(&exemplos_d, sizeof(exemplo) * quant_exemp);
		if (exemplos_h == NULL || exemplos_d == NULL){
			printf("\nErro de alcocacao!");
			system("pause");
		}
		processaExemplos(atrib_h, quant_atrib, exemplos_h, input);
		cudaMemcpy(exemplos_d, exemplos_h, sizeof(exemplo) * quant_exemp, cudaMemcpyHostToDevice);

		//carrega exemplos de teste do classificador final (apenas no host)
		int quant_exemp_test = contaExemplos(teste);
		exemplo *testes = (exemplo*)calloc(quant_exemp_test, sizeof(exemplo));
		processaExemplos(atrib_h, quant_atrib, testes, teste);

		//carrega enxames iniciais
		particula *enxames_h, *enxames_d;																								//ponteiros para alocação dinâmica do vetor com as partículas dos enxames
		int quant_total_particulas = h_param.quant_particulas * h_param.quant_enxames;
		cudaMallocHost((void**)&enxames_h, sizeof(particula) * quant_total_particulas);
		cudaMalloc(&enxames_d, sizeof(particula) * quant_total_particulas);
		if (enxames_h == NULL || enxames_d == NULL){
			printf("\nErro de alcocacao!");
			system("pause");
		}

		for (int i = 0; i < h_param.quant_enxames; i++){	//cria enxames individualmente e atribui a variável que apontará para todos os enxames
			particula *enxame_h = criaEnxame(h_param, atrib_h, quant_atrib);
			for (int j = 0; j < h_param.quant_particulas; j++)
				enxames_h[i * h_param.quant_particulas + j] = enxame_h[j];
		}

		for (int i = 0; i < quant_total_particulas; i++)
			enxames_h[i].velocidade = carregaVelocidadeParticulaDevice(enxames_h[i], atrib_h, quant_atrib);	//partículas dos enxames passam a apontar para velocidades alocadas no device

		cudaMemcpy(enxames_d, enxames_h, sizeof(particula) * quant_total_particulas, cudaMemcpyHostToDevice);

		//configura vetores que receberão as posições finais das partículas após a execução do kernel
		regra *posicoes_h, *posicoes_d;
		cudaMallocHost((void**)&posicoes_h, sizeof(regra) * quant_total_particulas);
		cudaMalloc(&posicoes_d, sizeof(regra) * quant_total_particulas);
		cudaMemcpy(posicoes_d, posicoes_h, sizeof(regra) * quant_total_particulas, cudaMemcpyHostToDevice);

		//CONFIGURA ALEATORIEDADE

		//configura aleatoriedade no host
		srand((unsigned)time(NULL));

		//configura aleatoriedade no device
		curandState* devStates;
		cudaMalloc(&devStates, quant_total_particulas * sizeof(curandState));
		setup_kernel << < h_param.quant_enxames, h_param.quant_particulas >> > (devStates, time(NULL));

		//INICIALIZA TIMERS

		//declara marcadores de tempo de início e fim
		cudaEvent_t eventoInicio, eventoFim;

		//inicializa marcadores de tempo
		cudaEventCreate(&eventoInicio);
		cudaEventCreate(&eventoFim);

		//INICIA PSO

		printf("Inicio de processamento no kernel\n");
		cudaEventRecord(eventoInicio);

		SPSOMultiEnxame << <h_param.quant_enxames, h_param.quant_particulas, h_param.quant_particulas * sizeof(particula) >> >
			(d_param, exemplos_d, quant_exemp, atrib_d, quant_atrib, output, enxames_d, posicoes_d, devStates);

		cudaDeviceSynchronize();
		cudaEventRecord(eventoFim);
		printf("Fim de processamento no kernel\n");

		//TRATA POSIÇÃO FINAL DAS PARTÍCULAS

		cudaMemcpy(posicoes_h, posicoes_d, sizeof(regra) * quant_total_particulas, cudaMemcpyDeviceToHost);
		posicoes_h = apagaRegrasIguais(posicoes_h, &quant_total_particulas, quant_atrib);
		regiao_pareto pareto = dominanciaDePareto(h_param, posicoes_h, quant_total_particulas);
		removeDominadas(&pareto);
		insereSolucoesEmArquivo(pareto, h_param, output);
		insereNomesSolucoesArquivo(pareto, h_param, output, atrib_h, quant_atrib);
		imprimeDominioPareto(pareto, quant_atrib);

		//INICIA PROCESSO DE AVALIAÇÃO DO CLASSIFICADOR

		//imprimeExemplosInt(testes, quant_exemp_test);

		classificador c = inicializaClassificador(pareto.solucoes, pareto.quant_sol_pareto);
		ordenaRegrasMaiorEspecificidade(c.regras, c.quant_regras);
		classificaExemplos(&c, testes, quant_exemp_test, quant_atrib);

		//imprime matriz no arquivo final
		for (int i = 0; i < quant_mat_conf; i++){
			printf("\nMatConf[%d]: %d", i, c.mat_conf[i]);
			fprintf(outputMatrizConf, "MatConf[%d]: %d\n", i, c.mat_conf[i]);
		}
		fprintf(outputMatrizConf, "\n", i, c.mat_conf[i]);

		//ARMAZENA TEMPO DE EXECUÇÃO MEDIDO

		float tempo = 0;
		cudaEventElapsedTime(&tempo, eventoInicio, eventoFim);
		fprintf(outputTempo, "%fms\n", tempo);

		//fecha arquivos desta partição e limpa memória cuda.
		fclose(input);
		fclose(output);

		cudaFree(atrib_h);
		cudaFree(exemplos_h);
		cudaFree(enxames_h);

		cudaFree(atrib_d);
		cudaFree(exemplos_d);
		cudaFree(enxames_d);
		cudaFree(devStates);
	}


	//FIM DO PROGRAMA

	fclose(arq_parametros);
	fclose(outputMatrizConf);
	fclose(outputTempo);
	system("pause");

	return 0;
}
