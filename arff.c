// Michel Conrado Cardoso Meneses (30/10/2014 - 16:57h)

#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>

// DEFINI��O DE CONSTANTES

#define quant_val 80																												// quantidade m�xima de valores discretos assumidos por um atributo;
#define tam_val 60																													// tamanho m�ximo do nome de um valor;
#define quant_max_atrib 100																											// quantidade m�xima de atributos do arquivo;
#define quant_mat_cont 8																											// quantidade de posi��es no vetor da matriz de contig�ncia;
#define quant_func_ob 8																												// quantidade de posi��es no vetor das fun��es objetivo;

/* posi��es da matriz de conting�ncia */
#define BH 0																														// B verdade e H verdade;
#define _BH 1																														// B falso e H verdade;
#define B_H 2																														// B verdade e H falso;
#define _B_H 3																														// B falso e H falso;
#define B 4																															// B verdade;
#define _B 5																														// B falso;
#define H 6																															// H verdade;
#define _H 7																														// H falso;

/* �ndices das fun��es objetivo */
#define ACC 0																														// precis�o;
#define ERR 1																														// erro;
#define NEGREL 2																													// confian�a negativa;
#define ACCLP 3																														// precis�o de Laplace;
#define SENS 4																														// sensitividade;
#define SPEC 5																														// cobertura;
#define COV 6																														// suporte;
#define SUP 7

const double MAX_PHI = 1;
const double MAX_OMEGA = 0.8;
const int INFINITO = 1000000;

// DEFINI��O DE ESTRUTURAS

struct atributo{
	
	char* nome;																													// nome do atributo
	int cod [quant_val];																										// c�digo num�rico dos valores do atributo;
	char valor [quant_val][tam_val];																							// valores do atributo;
	int quant_real;																												// quantidade real de valores que um atributo assume;
	int numerico;																												// se atributo for num�rico a flag ser� 1;
};
typedef struct atributo atributo;

struct exemplo{
	
	int campos [quant_val];																										// vetor de atributos do exemplo;
	int quant_real;																												// quantidade real de atributos do exemplo;
};
typedef struct exemplo exemplo;

struct regra{
	
	int valores[quant_max_atrib];
	int mat_cont [quant_mat_cont];																								// vetor da matriz de conting�ncia;
	double func_ob [quant_func_ob];																								// vetor contendo o valor das fun��es objetivo da regra;
	int nula;																													// booleano que indica se a regra � nula (1) ou n�o (0);
	double crowding_distance;
	int quant_dominadores;																										// quantidade de regras que dominam por Pareto esta regra;
};
typedef struct regra regra;

struct parametros{
	
	char* arquivo;																												// diret�rio do arquivo de treino .arff;
	int execucoes;																												// n�mero de execu��es do algoritmo;
	int classe;																													// define se regras geradas ser�o para classe positiva (1), negativa (0) ou ambas (-1);	
	int funcao_obj;																												// fun��o objetivo a ser usada na B.L;	
	int bl_interacoes;																											// n� de intera��es do algoritmo de busca local;
	int bl_vizinhos;																											// n� de vizinhos a serem gerados em cada intera��o do algoritmo de busca local;
	char* funcoes_obj_pareto;																									// sequ�ncia de bits que indicam as fun��es objetivos a serem consideradas na domin�ncia de Pareto;
	int quant_regras_pareto;																									// n� de regras que ser�o geradas e analisadas quanto � domin�ncia de pareto; 
	int quant_particulas;																										// quantidade de part�culas de um enxame;
	int tamanho_arquivo;																										// quantidade m�xima de solu��es do S-PSO
	int metodo_dopagem_solucao;																									// forma de inserir regras dominadas temporariamente no arquivo solu��o;
	int metodo_gerar_regras;																									// forma de gerar regras aleat�rias. Gerar regras uniformimente aleat�rias = 0, privilegiar o valor "_" = 1 (caso seja 1, atribuir um valor ao campo "prob_valor_vazio");
	int prob_valor_vazio;																										// probabilidade do valor vazio ser atribu�do a um atributo de uma regra gerada aleatoriamente (s� � usado caso o valor do campo "metodo_gerar_regras" seja 1);
};
typedef struct parametros parametros;

struct regiao_pareto{
	
	regra* solucoes;																											// conjunto de regras n�o dominadas;
	int func_obj [quant_func_ob];																								// fun��es objetivos consideradas para verifica��o de domin�ncia;
	int quant_sol_pareto;																										// quantidade de regras contidas no conjunto n�o dominado de pareto;
};
typedef struct regiao_pareto regiao_pareto;

struct particula{
	
	regra posicao;
	regra lBest;
	regra gBest;
	double** velocidade;
};
typedef struct particula particula;

// DECLARA��O DE FUN��ES

/* Fun��o para carrear arquivo .arff */
FILE* carregarArq (char argv[]){
	
	FILE* input = fopen(argv, "r");
	
	if (input == NULL){																											// caso o  arquivo n�o exista
		
		printf("Arquivo n�o encontrado!\n");																					// emite mensagem de erro
		printf("\n%s", argv);
		printf("\nlenght = %d", strlen(argv));
		exit(1);																												// e finaliza
	}
	return input;
}

/* Fun��o para contar atributos da classe */
int contaAtributos (FILE* input){
	
	int quant_atrib = 0;																										// quantidade de atributos dos elementos da base de dados;
	
	while(!feof(input)){
		
		const tam_str_aux = 1000;																								// define o tamanho m�ximo de uma linha a ser lida;	
		char str_aux [tam_str_aux];																								// string auxiliar que receber� a linha do arquivo .arff a ser lida;
		fgets(str_aux, tam_str_aux, input);																						// l� linha do arquivo;
		
		char* pont = strstr(str_aux, "@attribute");																				// busca pela primeira ocorr�ncia da palavra "@attribute " na linha;
		
		if(pont != NULL){																										// caso a palavra seja encontrada
			quant_atrib++;																										// incrementa n�mero de atributos;
		}
	}
	
	rewind(input);																												// reposiciona ponteiro no in�cio do arquivo;
	return quant_atrib;
}

/* Fun��o que gera uma nova estrutura "atributo" j� inicializada */
atributo inicializaAtributo (){
	
	atributo atr;
	atr.quant_real = 0;
	atr.numerico = 0;
	atr.quant_real = 0;
	
	return atr;
}

void setNomeAtributo(char* string, atributo* atr){
	
	char* str = strchr(string, ' ');
	str = str + 1;					// desloca um caracter � direita
	int cont = 0;
	
	if(str != NULL){
		
		while(str[cont] != ' ')
			cont++;
		
		(*atr).nome = (char*) calloc(cont + 1, sizeof(char));
		strncpy((*atr).nome, str, cont);
		(*atr).nome[cont] = '\0';
	}
}

/* Mapeia os valores de um atributo */
atributo mapeiaAtributo (char* string){
	
	atributo atr = inicializaAtributo();
	
	char* num = strstr(string, "numeric");
	
	if(num != NULL){																											// caso encontre-se a palavra reservada "numeric"
		atr.numerico = 1;																										// o atributo � marcado como num�rico;
	}
	else{
		
		setNomeAtributo(string, &atr);
		//printf("\nAqui:%s", atr.nome);
		//printf("\nSize:%d", strlen(atr.nome));
		char* str = strchr(string, '{');																						// posiciona ponteiro na primeira ocorr�ncia de '{';
		int indice = 1;																											// posicao do caracter acessado na string "str";
		int cont_valor = 0;																										// representa a contagem dos valores que foram mapeados;
		//printf("lenght = %d\n", strlen(str));
		while((indice + 1) < (strlen(str) - 1) ){																				// enquanto o �ltimo caracter ('}') n�o for lido;
			
			int indice_valor = 0;																								// representa a posi��o do caractere do valor;
			while( (str[indice] != ',') && ((indice + 1) < (strlen(str) - 1) )){
				//printf("(indice + 1) = %d ", (indice + 1));
				//printf("atr.valor[%d][%d] = %c\n", cont_valor, indice_valor,str[indice]);
				atr.valor[cont_valor][indice_valor] = str[indice];																
				//printf("atr.quant_real = %d\n", atr.quant_real);
				indice_valor++;
				indice++;				
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
void processaAtributos (atributo* atrib, int quant_atrib, FILE* input){

	int cont = quant_atrib;																										// define contador a ser decrementado;
	const tam_str_aux = 1000;																									// define o tamanho m�ximo de uma linha a ser lida;
	
	while(!feof(input)){
		
		char str_aux [tam_str_aux];																								// string auxiliar que receber� a linha do arquivo .arff a ser lida;
		fgets(str_aux, tam_str_aux, input);																						// l� linha do arquivo;
		
		char* pont = strstr(str_aux, "@attribute");																				// busca pela primeira ocorr�ncia da palavra "@attribute " na linha;
		
		if(pont != NULL){																										// caso a palavra seja encontrada
			*(atrib + quant_atrib - cont) = mapeiaAtributo(pont);																// insere no vetor um novo atributo mapeado;
			cont--;																												// decrementa contador;
		}
	}
	
	rewind(input);
}

/* Imprime atributos e todos os seus poss�veis valores */
void imprimeAtributos(atributo* atrib, int quant_atrib){
	
	printf("\n\n\t\t\t\t\t\t\t#LISTA DE ATRIBUTOS#\n\n");
	int i;
	for(i = 0; i < quant_atrib; i++){
		
		printf("-> Atributo %d: %s\n\n", i, (*(atrib + i)).nome);
		if( (*(atrib + i)).numerico == 0 ){
			int j;
			for(j = 0; j < (*(atrib + i)).quant_real; j++){
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
		
		if(indice_valor == -1)			
			return "___";
		else
			return atrib[indice_atrib].valor[indice_valor];
	}
	else if (indice_atrib >= quant_atrib){
		
		printf("\nIndice de atributo invalido!");
		exit(1);
	}
	else if (indice_valor >= atrib[indice_atrib].quant_real){
		
		printf("\nIndice de valor invalido!");
		printf("indice de valor = %d", indice_valor);
		exit(1);
	}
}

/* Conta o n�mero de exemplos da base de dados */
int contaExemplos(FILE* input){
	
	const tam_str_aux = 1000;																									// define o tamanho m�ximo de uma linha a ser lida;
	char* pont;																													
	int contagem = 1;
	
	do{
		char str_aux [tam_str_aux];																								// string auxiliar que receber� a linha do arquivo .arff a ser lida;
		fgets(str_aux, tam_str_aux, input);																						// l� linha do arquivo;
		pont = strstr(str_aux, "@data");																						// busca pela primeira ocorr�ncia da palavra "@attribute " na linha;
		
		}while(pont == NULL && (!feof(input)));
	
	char str_aux [tam_str_aux];																									// string auxiliar que receber� a linha do arquivo;
	fgets(str_aux, tam_str_aux, input);																							// l� linha do arquivo e pula para a pr�xima;	
	
	while(!feof(input)){
		
		char str_aux [tam_str_aux];																								// string auxiliar que receber� a linha do arquivo;
		fgets(str_aux, tam_str_aux, input);																						// l� linha do arquivo e pula para a pr�xima;																				// pula para a pr�xima linha;
		
		contagem++;
	}
	
	rewind(input);																												// reposiciona ponteiro no in�cio do arquivo;
	free(pont);
	return contagem;
}

/* Retorna um exemplo inicializado */
exemplo inicializaExemplo(int quant_atrib){
	
	exemplo exemp;
	exemp.quant_real = quant_atrib;
	return exemp;
}

int valorInt (atributo* atrib, int posicao, char* string){
	
	int igual = 0;																												// flag que indica se a string passada � igual a algum valor no vetor de atributos;
	int cod;
	int i;
	
	for(i = 0; i < (*(atrib + posicao)).quant_real; i++ ){
		
		if( strcmp( (*(atrib + posicao)).valor[i], string ) == 0 ){
			igual = 1;
			cod = (*(atrib + posicao)).cod[i];
		}
	}
	
	if(igual == 0){
		cod = -1;
		//printf("String = %s\n", string);
	}
	
	return cod;
}

/* Converte exemplo em vetor de inteiros */
exemplo converteExemplo(atributo* atrib, int quant_atrib, char* linha){
	
	exemplo exemp = inicializaExemplo(quant_atrib);																				// cria��o de um novo exemplo;
	int atrib_atual = 0;																										// atributo atual sendo lido;
	int cod;																													// c�digo do atributo;
	char* pont;
	pont = strtok(linha, ",\n");
	
	while(pont != NULL){
		
		if( strcmp(pont, "") == 0 || strcmp(pont, "?") == 0 ){																	//se o valor do exemplo for vazio ou ? ent�o seu c�digo ser� -1;
			exemp.campos[atrib_atual] = -1;
		}
		else{
			cod = valorInt(atrib, atrib_atual, pont);
			if( cod == -1 ){																									// se o valor do exemplo n�o faz parte dos atributos o sistema � encerrado;
				printf("ERRO: Valor de exemplo inexistente!\n");
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
	const tam_str_aux = 1000;																									// define o tamanho m�ximo de uma linha a ser lida;
	char* pont;																													
	 
	do{
		char str_aux [tam_str_aux];																								// string auxiliar que receber� a linha do arquivo .arff a ser lida;
		fgets(str_aux, tam_str_aux, input);																						// l� linha do arquivo;
		pont = strstr(str_aux, "@data");																						// busca pela primeira ocorr�ncia da palavra "@attribute " na linha;
		
		}while(pont == NULL && (!feof(input)));
	
	while(!feof(input)){
		
		char str_aux [tam_str_aux];																								// string auxiliar que receber� a linha do arquivo .arff a ser lida;
		fgets(str_aux, tam_str_aux, input);																						// l� linha do arquivo;
		char* pont = strstr(str_aux, "");																				
		*(exemplos + quant_atrib - cont) = converteExemplo(atrib, quant_atrib, pont);
		cont--;
	}
	
	free(pont);
}

void imprimeExemplosInt (exemplo* exemplos, int quant_exemp){
	
	printf("\n\n\t\t\t\t\t\t#LISTA DOS VETORES DE EXEMPLOS#\n\n");
	
	int i,j;
	for(i = 0; i < quant_exemp; i++){
		printf("Exemplo %2d: ", i);
		
		for(j=0; j < (*(exemplos + i)).quant_real - 1; j++){
			printf("%d, ", (*(exemplos + i)).campos[j]);
		}
		
		printf("%d\n\n", (*(exemplos + i)).campos[(*(exemplos + i)).quant_real - 1]);
	}
	
	printf("*--------------------------------------------------------------------------------------------------------*");
	printf("\n\n\n");
}

regra inicializaRegra (){
	
	regra r;
	int i;
	
	for(i = 0; i < quant_mat_cont; i++)																						// inicializa com 0's a matriz de contig�ncia;
		r.mat_cont[i] = 0;
	
	r.nula = 1;																												// regra inicialmete � nula;
	r.crowding_distance = 0;
	r.quant_dominadores = 0;
	
	return r;
}

/* Gera uma regra aleat�ria */
regra geraRegra (parametros param, atributo* atrib, int quant_atrib, int classe){

	regra r = inicializaRegra();
	int i, limiar;
	
	//srand( (unsigned) time(NULL) );
	for(i = 0; i < quant_atrib - 1; i++){
		
		if(param.metodo_gerar_regras == 1){
			
			limiar = rand()%101;
			if(param.prob_valor_vazio >= limiar)
				r.valores[i] = -1;
			else 
				r.valores[i] = rand()%( (*(atrib + i)).quant_real);
		}
		else if(param.metodo_gerar_regras == 0)
			r.valores[i] = rand()%( (*(atrib + i)).quant_real + 1) - 1;
	}
	
	switch( classe ){
	
		case 0:
			r.valores[quant_atrib - 1] = 0;
		break;
		
		case 1:
			r.valores[quant_atrib - 1] = 1;
		break;
		
		default:
			r.valores[quant_atrib - 1] = rand()%( (*(atrib + i)).quant_real);												// valor aleat�rio para a classe;
	}
	
	r.nula = 0;																												// regra deixa de ser nula;
	
	return r;
}

void imprimeRegra(regra r, int quant_atrib){
	
	//printf("\nRegra: ");
	
	int i;
	for(i = 0; i < quant_atrib - 1; i++){
		
		printf("%d ", r.valores[i]);
	}
	
	printf("-> %d\n", r.valores[quant_atrib - 1]);
}

void imprimeRegraCompleta(regra r, int quant_atrib){
	
	//printf("\nRegra: ");
	
	int i;
	for(i = 0; i < quant_atrib - 1; i++){
		
		printf("%d ", r.valores[i]);
	}
	
	printf("-> %d\n", r.valores[quant_atrib - 1]);
	
	printf("\n\tCrowding Distance: %f", r.crowding_distance);
	printf("\n\tQuantidade de dominadores: %d\n\n", r.quant_dominadores);
}

void imprimeNomesRegra(regra r, atributo* atrib, int quant_atrib){
	
	int i;
	for(i = 0; i < quant_atrib - 1; i++){
		
		printf("%s ", getNomeValorAtributo(atrib, quant_atrib, i, r.valores[i]));
	}
	
	printf("-> %s\n", getNomeValorAtributo(atrib, quant_atrib, quant_atrib - 1, r.valores[quant_atrib - 1]));
}

/* Zera todas as posi��es da matriz de conting�ncia de uma regra */
void zeraMatrizCont( regra* r ){
	
	int i;
	
	for( i = 0; i < quant_mat_cont; i++ ){
		
		(*r).mat_cont[ i ] = 0;
	}
}

void preencheMatrizCont (regra* r, exemplo* exemplos, int quant_exemp, int quant_atrib){
	
	zeraMatrizCont(r);
	int i,j,b,h;
	
	for(i = 0; i < quant_exemp; i++){
		
		b = 1;
		h = 1;
		
		for(j = 0; j < quant_atrib - 1; j++){
			
			if( ( (*(exemplos + i)).campos[j]  != (*r).valores[j] ) && ( (*r).valores[j]  != -1 ) ){
				b = 0;
			}
		}
		
		if( (*(exemplos + i)).campos[ quant_atrib - 1 ]  != (*r).valores[ quant_atrib - 1 ] ){
			h = 0;
		}
		
		if( b == 1 )
			(*r).mat_cont[B]++;
		else
			(*r).mat_cont[_B]++;
		
		if( h == 1 )
			(*r).mat_cont[H]++;
		else
			(*r).mat_cont[_H]++;
		
		if( b == 1 && h == 1 )
			(*r).mat_cont[BH]++;
		
		else if (b == 1 && h == 0 )
			(*r).mat_cont[B_H]++;
		
		else if (b == 0 && h == 1 )
			(*r).mat_cont[_BH]++;

		else if (b == 0 && h == 0 )
			(*r).mat_cont[_B_H]++;	
	}
}

void imprimeMatrizCont (regra r){
	
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

void calculaFuncoesObj ( regra* r, int quant_atrib, atributo* atrib, int quant_exemp ){
	
	int quant_classes = (*(atrib + quant_atrib - 1)).quant_real;
	
	if( (*r).mat_cont[B] != 0 )
		(*r).func_ob[ACC] = (double) ( (*r).mat_cont[BH] ) / (*r).mat_cont[B];
	else
		(*r).func_ob[ACC] = -1;
	if( (*r).mat_cont[B] != 0 )
		(*r).func_ob[ERR] = (double) ( (*r).mat_cont[B_H] ) / (*r).mat_cont[B];
	else
		(*r).func_ob[ERR] = -1;
	if( (*r).mat_cont[_B] != 0 )
		(*r).func_ob[NEGREL] = (double) ( (*r).mat_cont[_B_H] ) / (*r).mat_cont[_B];
	else
		(*r).func_ob[NEGREL] = -1;
	if( ( (*r).mat_cont[B] + quant_classes ) != 0 )
		(*r).func_ob[ACCLP] = (double) ( (*r).mat_cont[BH] + 1 ) / ( (*r).mat_cont[B] + quant_classes );
	else
		(*r).func_ob[ACCLP] = -1;
	if( (*r).mat_cont[H] != 0 )
		(*r).func_ob[SENS] = (double) ( (*r).mat_cont[BH] ) / (*r).mat_cont[H];
	else
		(*r).func_ob[SENS] = -1;
	if( (*r).mat_cont[_H] != 0 )
		(*r).func_ob[SPEC] = (double) ( (*r).mat_cont[_B_H] ) / (*r).mat_cont[_H];
	else
		(*r).func_ob[SPEC] = -1;
	if( quant_exemp != 0 )
		(*r).func_ob[COV] = (double) ( (*r).mat_cont[B] ) / quant_exemp;
	else
		(*r).func_ob[COV] = -1;
	if( quant_exemp != 0 )
		(*r).func_ob[SUP] = (double) ( (*r).mat_cont[BH] ) / quant_exemp;
	else
		(*r).func_ob[SUP] = -1;	
}

void imprimeFuncoesObj( regra r ){
	
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

regra* geraRegras( parametros param, int quant_regras, atributo* atrib, int quant_atrib, int classe, exemplo* exemplos, int quant_exemp ){
	
	int i;
	regra* regras = (regra*) calloc(quant_regras, sizeof( regra ));
	if(regras == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	for(i = 0; i < quant_regras; i++){
		
		*(regras + i) = geraRegra(param, atrib, quant_atrib, classe);
		preencheMatrizCont((regras + i), exemplos, quant_exemp, quant_atrib);
		calculaFuncoesObj((regras + i), quant_atrib, atrib, quant_exemp);
		//imprimeRegra(*(regras + i), quant_atrib);
		//imprimeMatrizCont(*(regras + i));
		//imprimeFuncoesObj(*(regras + i));
	}
	
	return regras;
}

/* Dado uma string, a fun��o retorna o restante da linha de um arquivo ap�s tal string, somente quando a linha come�a pela string*/
char* saltaStringArq ( char* string, FILE* arq ){
	
	const tam_str_aux = 200;																								// define o tamanho m�ximo de uma linha a ser lida;
	char str_aux [tam_str_aux];																								// string auxiliar que receber� a linha do arquivo .arff a ser lida;
	
	fgets(str_aux, tam_str_aux, arq);																						// l� linha do arquivo;
	char* pont = (char*) calloc( tam_str_aux, sizeof(char) );																// aloca espa�o de mem�ria equivalente ao tamanho m�ximo da linha lida;
	if(pont == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	strncpy(pont, str_aux, tam_str_aux);																					// copia a linha lida para o ponteiro "pont";
	pont = (pont + strlen(string));																							// armazena no ponteiro "pont" o resto da linha ap�s "string";
	return pont;
}

/* Dado uma string, a fun��o retorna o restante da linha de um arquivo ap�s tal string, independentemente da posi��o da string*/
char* localizaString(char* string, FILE* arq){
	
	const tam_str_aux = 200;																								// define o tamanho m�ximo de uma linha a ser lida;
	char str_aux [tam_str_aux];																								// string auxiliar que receber� a linha do arquivo .arff a ser lida;
	
	fgets(str_aux, tam_str_aux, arq);																						// l� linha do arquivo;
	char* pont = (char*) calloc( tam_str_aux, sizeof(char) );																// aloca espa�o de mem�ria equivalente ao tamanho m�ximo da linha lida;
	if(pont == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	strncpy(pont, str_aux, tam_str_aux);																					// copia a linha lida para o ponteiro "pont";
	
	int indice_linha = 0, indice_string = 0;
	int flag = 0;
	
	while(indice_linha < strlen(pont) && flag == 0){
		while(indice_string < strlen(string) && indice_linha + indice_string < strlen(pont) & flag == 0){
			if(string[indice_string] == pont[indice_linha + indice_string]){
				indice_string++;
				if(indice_string == strlen(string))
					flag = 1;
			}
			else{
				indice_linha++;
				indice_string = 0;
			}
		}
	}
	
	if(flag == 1){
		pont = (pont + indice_linha + strlen(string));
	}
	else
		return NULL;
	
	return pont;
}

void carregaParametros( FILE* file, parametros* param ){
	
	(*param).arquivo = strtok( saltaStringArq("@arquivo_treino:", file) , "\n");
	(*param).execucoes = atoi( saltaStringArq("@execucoes:", file) );
	(*param).classe = atoi( saltaStringArq("@classe:", file) );
	(*param).funcao_obj = atoi( saltaStringArq("@funcao_objetivo:", file) );
	(*param).bl_interacoes = atoi( saltaStringArq("@num_inter_bl:", file) );
	(*param).bl_vizinhos = atoi( saltaStringArq("@num_vizinhos_bl:", file) );
	(*param).funcoes_obj_pareto =  strtok( saltaStringArq("@dominancia_de_pareto:", file) , "\n");
	(*param).quant_regras_pareto = atoi( saltaStringArq("@quant_regras_pareto:", file) );
	(*param).quant_particulas = atoi( saltaStringArq("@quant_particulas:", file) );
	(*param).tamanho_arquivo = atoi( saltaStringArq("@tamanho_arquivo:", file) );
	(*param).metodo_dopagem_solucao = atoi( saltaStringArq("@metodo_dopagem_solucao:", file) );
	(*param).metodo_gerar_regras = atoi( saltaStringArq("@metodo_gerar_regras:", file) );
	(*param).prob_valor_vazio = atoi( saltaStringArq("@prob_valor_vazio:", file) );
	
	rewind(file);
}

void imprimeParametros ( parametros param ){
	
	printf("\nParametros:\n\n");
	printf("Arquivo = %s\n", param.arquivo);
	printf("Execucoes = %d\n", param.execucoes);
	printf("Classe = %d\n", param.classe);
	printf("Funcao Objetivo = %d\n", param.funcao_obj);
	printf("Numero de interacoes = %d\n", param.bl_interacoes);
	printf("Numero de vizinhos = %d\n", param.bl_vizinhos);
	printf("Funcoes objetivo de Pareto = %s\n", param.funcoes_obj_pareto);
	printf("Quantidade de regras de Pareto = %d\n", param.quant_regras_pareto);
	printf("Quantidade de particulas = %d\n", param.quant_particulas);
	printf("Tamanho maximo do arquivo = %d\n", param.tamanho_arquivo);
	printf("M�todo de dopagem de solucao = %d\n", param.metodo_dopagem_solucao);
	printf("M�todo de gera��o de regras = %d\n", param.metodo_gerar_regras);
	printf("Probabilidade de gerar atributo vazio (_) = %d\n", param.prob_valor_vazio);
}

/* Altera aleatoriamente um atributo de uma certa regra passada como par�metro */
regra alteraRegra( regra base, atributo* atrib, int quant_atrib ){
	
	regra nova = base;
	int indice = rand()%(quant_atrib - 1);																					// indice aleat�rio do atributo a ser alterado;
	nova.valores[ indice ] = rand()%( (*(atrib + indice)).quant_real + 1) - 1;
	
	printf("\n\nBase = :");
	imprimeRegra(base, quant_atrib);
	printf("\n\nNova = :");
	imprimeRegra(nova, quant_atrib);
	
	return nova;
}

/* Fun��o que retorna 0 caso a 1� regra seja melhor ou igual que a 2� regra  */
int comparaRegras ( regra r0, regra r1, int func_ob ){
	
	if( func_ob == ERR){
		if( r0.func_ob[ERR] <= r1.func_ob[ERR] )
			return 0;
		else
			return 1;
	}
	else{
		if( r0.func_ob[func_ob] >= r1.func_ob[func_ob] )
			return 0;
		else
			return 1;
	}
}

void buscaLocal( parametros param, atributo* atrib, int quant_atrib, exemplo* exemplos, int quant_exemp ){
	
	regra* vizinhos = (regra*) calloc( param.bl_vizinhos, sizeof(regra) );
	if(vizinhos == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	regra atual, vizinho;
	int i,j,sair;																											// caso durante uma intera��o nenhum vizinho for melhor que a regra atual o algoritmo � encerrado;
	
	atual = geraRegra(param, atrib, quant_atrib, param.classe);
	preencheMatrizCont(&atual, exemplos, quant_exemp, quant_atrib);
	calculaFuncoesObj(&atual, quant_atrib, atrib, quant_exemp);
	
	for( i = 0; i < param.bl_interacoes; i++ ){
		
		sair = 1;																											// inicialmente deve sair;
		
		for( j = 0; j < param.bl_vizinhos; j++ ){
			
			*(vizinhos + j) = alteraRegra(atual, atrib, quant_atrib);
			preencheMatrizCont( (vizinhos + j), exemplos, quant_exemp, quant_atrib);
			calculaFuncoesObj( (vizinhos + j), quant_atrib, atrib, quant_exemp);
			
			if( comparaRegras( atual, *(vizinhos + j), param.funcao_obj ) == 1  ){
				
				atual = *(vizinhos + j);
				sair = 0;																									// sa�da adiada;
			}
			
		}
		
		if( sair == 1 ){
			
			printf("\nNenhum vizinho melhor que a regra atual!");
			break;
		}
	}
	
	if( sair == 0 )
		printf("\nLimite de interacoes ultrapassado!");
	
	printf("\nNumero de interacoes realizadas = %d", i + 1);
	imprimeRegra(atual, quant_atrib);
	imprimeMatrizCont(atual);
	imprimeFuncoesObj(atual);
}

//////////////////////////////////////// IN�CIO DOMIN�NCIA DE PARETO //////////////////////////

/*Retorna -1 se r1 domina r2, 0 se nenhuma das duas regras domina a outra, 1 se r2 domina r1*/
int dominaPorPareto ( regra r1, regra r2, regiao_pareto pareto ){
	
	int i;
	int saida = 0;
	
	for(i = 0; i < quant_func_ob; i++){
		if( pareto.func_obj[i] == 1 ){
			if( r1.func_ob[i] > r2.func_ob[i] ){
				
				if( saida == 0 || saida == -1 )
					saida = -1;
				else
					return 0;
			}
			if( r1.func_ob[i] < r2.func_ob[i] ){
				
				if( saida == 0 || saida == 1 )
					saida = 1;
				else
					return 0;		
			}
		}
	}
	
	return saida;
}

/*Cria e inicializa uma vari�vel do tipo "regiao_pareto"*/
regiao_pareto inicializaPareto(parametros param){
	
	regiao_pareto pareto;
	
	pareto.solucoes = (regra*) calloc(1, sizeof(regra));
	if(pareto.solucoes == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	pareto.solucoes[0].nula = 1;
	pareto.quant_sol_pareto = 1;
	
	int i;
	for(i = 0; i < quant_func_ob; i++){
		pareto.func_obj[i] = ( param.funcoes_obj_pareto[i] ) - 48;
	}
	
	return pareto; 	
}

void inserePareto(regiao_pareto* pareto, regra r){
	
	int i, inserido = 0;
	
	for(i = 0; i < (*pareto).quant_sol_pareto; i++){
		
		if( (*pareto).solucoes[i].nula == 1 ){
			
			(*pareto).solucoes[i] = r;
			(*pareto).solucoes[i].nula = 0;
			inserido = 1;
			break;
		}
	}
	
	if(inserido == 0){
		
		(*pareto).solucoes = (regra*) realloc( (*pareto).solucoes, ( (*pareto).quant_sol_pareto + 1)*sizeof(regra) );
		(*pareto).quant_sol_pareto++;
		(*pareto).solucoes[(*pareto).quant_sol_pareto - 1] = r;
		(*pareto).solucoes[(*pareto).quant_sol_pareto - 1].nula = 0;
	}
}

int contaRegrasNaoNulasPareto(regiao_pareto pareto){
	
	int cont = 0;
	int i;
	
	for(i = 0; i < pareto.quant_sol_pareto; i++){
		
		if(pareto.solucoes[i].nula == 0)
			cont++;
	}
	
	return cont;
}

void apagaSolucoesNulasPareto(regiao_pareto* pareto){
	
	int quant_sol_nao_nulas = contaRegrasNaoNulasPareto(*pareto);
	regra* solucoes = (regra*) calloc(quant_sol_nao_nulas, sizeof(regra));
	if(solucoes == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	int cont = 0;
	int i;
	
	for(i = 0; i < (*pareto).quant_sol_pareto; i++){
		
		if((*pareto).solucoes[i].nula == 0){
			
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
	for(i = 0; i < pareto.quant_sol_pareto; i++){
		
		if( pareto.solucoes[i].nula == 0 )
			imprimeRegra(pareto.solucoes[i], quant_atrib);
	}
}

void imprimeDominioParetoComObjetivos(regiao_pareto pareto, int quant_atrib, parametros param){
	
	printf("\n#Dominio de Pareto:\n");
	
	int i,j;
	for(i = 0; i < pareto.quant_sol_pareto; i++){
		
		printf("\n");
		imprimeRegra(pareto.solucoes[i], quant_atrib);
		
		for(j = 0; j < quant_func_ob; j++){
			
			if(param.funcoes_obj_pareto[j] == '1'){
				imprimeFuncaoObj(pareto.solucoes[i], j);
			}
		}
	}	
}

void zeraQuantDominadoresRegras(regra* regras, int quant_regras){
	
	int i;
	for(i = 0; i < quant_regras; i++){
		
		regras[i].quant_dominadores = 0;
	}
}

regiao_pareto dominanciaDePareto(parametros param, regra* regras, int quant_regras){
	
	regiao_pareto pareto = inicializaPareto(param);
	zeraQuantDominadoresRegras(regras, quant_regras);
	int i,j;																										  
	//int dominada;																										// dominada = 0 se regra n�o � dominada por nenhuma outra regra dentro de "pareto";
	
	for(i = 0; i < quant_regras; i++){
		
		//dominada = 0;
		
		for(j = 0; j < pareto.quant_sol_pareto; j++){
			
			if( pareto.solucoes[j].nula == 0 ){
				
				if( dominaPorPareto(*(regras + i), pareto.solucoes[j], pareto) == -1 ){
					pareto.solucoes[j].nula = 1;
					pareto.solucoes[i].quant_dominadores++;
				}
				else if( dominaPorPareto(*(regras + i), pareto.solucoes[j], pareto) == 1 ){
					//dominada = 1;
					regras[i].quant_dominadores++;
					//printf("\n%d",regras[i].quant_dominadores);
				}
			}
		}
	
		if( /*dominada == 0*/ regras[i].quant_dominadores == 0 ){
			
			//printf("\ninserindo em pareto: %d", regras[i].quant_dominadores);
			inserePareto(&pareto, *(regras + i));
		}
	}
	
	apagaSolucoesNulasPareto(&pareto);
	return pareto;
}

//////////////////////////////////////// FIM DOMIN�NCIA DE PARETO /////////////////////////////

//////////////////////////////////////// IN�CIO S-PSO ////////////////////////////////////////

double** inicializaVelocidadeParticula(atributo* atrib, int quant_atrib, int indice){
	
	double** vel = (double**) calloc(quant_atrib, sizeof(double*));
	if(vel == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	int i,j;
	for(i = 0; i < quant_atrib; i++){
		
		vel[i] = (double*) calloc(atrib[i].quant_real + 1, sizeof(double));
		if(vel[i] == NULL){
		
			printf("\nErro de alcocacao!");
			exit(1);
		}
		for(j = 0; j < atrib[i].quant_real + 1; j++){
			
			vel[i][j] = (rand()%101)/100.0;
		}
	}
	
	return vel;
}

particula criaParticula(parametros param, atributo* atrib, int quant_atrib, int i){
	
	particula p;
	p.posicao = geraRegra(param, atrib, quant_atrib, param.classe);
	//imprimeRegra(p.posicao, quant_atrib);
	p.lBest = p.posicao;
	p.gBest = p.posicao;
	p.gBest.nula = 1;
	p.velocidade = inicializaVelocidadeParticula(atrib, quant_atrib, i);
	
	return p;
}

particula* criaEnxame(parametros param, atributo* atrib, int quant_atrib){
	
	particula* enxame = (particula*) calloc(param.quant_particulas, sizeof(particula));
	if(enxame == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	int i;
	for(i = 0; i < param.quant_particulas; i++){
		
		enxame[i] = criaParticula(param, atrib, quant_atrib, i);
		
	}
	
	return enxame;
}

void calculaObjetivosEnxame(particula* enxame, parametros param, exemplo* exemplos, int quant_exemp, atributo* atrib, int quant_atrib){
	
	int i;
	for(i = 0; i < param.quant_particulas; i++){
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
	
	int i,j;
	for(i = 0; i < quant_atrib; i++){
		
		printf("\n\tatributo %d", i + 1);
		for(j = 0; j < atrib[i].quant_real + 1; j++){
			
			printf("\n\t%20s - %.2f%%", getNomeValorAtributo(atrib, quant_atrib, i, j - 1), p.velocidade[i][j]);
		}
	}
}

void imprimeParticula(particula p, parametros param, atributo* atrib, int quant_atrib){
	
	if(p.posicao.nula == 0){
		
		printf("\n.Posicao = ");
		imprimeRegraCompleta(p.posicao, quant_atrib);
	}
	
	if(p.lBest.nula == 0){
		
		printf(".Melhor local = ");
		imprimeRegraCompleta(p.lBest, quant_atrib);
	}
	
	if(p.gBest.nula == 0){
		
		printf(".Melhor global = ");
		imprimeRegraCompleta(p.gBest, quant_atrib);
	}
	
	imprimeVelocidadeParticula(p,atrib, quant_atrib);
	
	printf("\n.Objetivos:");
	
	int i;
	for(i = 0; i < quant_func_ob; i++){
		
		if(param.funcoes_obj_pareto[i] == '1'){
			imprimeFuncaoObj(p.posicao, i);
		}
	}
}

void imprimeEnxame(particula* enxame, parametros param, atributo* atrib, int quant_atrib){
	
	printf("\n#Enxame:\n");
	
	int i;
	for(i = 0; i < param.quant_particulas; i++){
		
		printf("\nParticula %d:\n", i + 1);
		imprimeParticula(enxame[i], param, atrib, quant_atrib);
	}
}

regra* enxameParaRegras(parametros param, particula* enxame){
	
	regra* regras = (regra*) calloc(param.quant_particulas, sizeof(regra));
	if(regras == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	int i;
	for(i = 0; i < param.quant_particulas; i++){
		
		regras[i] = enxame[i].posicao;
	}
	
	return regras;
}

/*Dado dois vetores de regras v1 e v2 e seus respectivos tamanhos t1 e t2, 
retorna um novo vetor v de tamanho t1 + t2 contendo as regras de v1 e v2*/
regra* uneRegras(regra* regras1, int quant_regras1, regra* regras2, int quant_regras2){
	
	regra* regras = (regra*) calloc(quant_regras1 + quant_regras2, sizeof(regra));
	if(regras == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	int i;
	for(i = 0; i < quant_regras1; i++){
		
		if(regras1[i].nula != 1)
			regras[i] = regras1[i];
	}
	for(i = 0; i < quant_regras2; i++){
		
		if(regras2[i].nula != 1)
			regras[i + quant_regras1] = regras2[i];
	}
	
	return regras;
}

int verificaIgualdadeRegras(regra r1, regra r2, int quant_atrib){
	
	int iguais = 1;
	int i;
	
	for(i = 0; i < quant_atrib; i++){
		
		if(r1.valores[i] != r2.valores[i])
			iguais = 0;
	}
	
	return iguais;
}

regra* apagaRegrasIguais(regra* r, int* quant_regras, int quant_atrib){
	
	int i,j;
	
	for(i = 0; i < *quant_regras; i++){
		for(j = 0; j < *quant_regras; j++){
			
			if( (i != j) && (r[i].nula == 0) && (r[j].nula == 0) && (verificaIgualdadeRegras(r[i], r[j], quant_atrib) == 1) ){
				
				r[i].nula = 1;
			}
		}
	}
	
	int cont = 0;
	
	for(i = 0; i < *quant_regras; i++){
		
		if(r[i].nula == 0)
			cont++;
	}
	//printf("\n%d %d", *quant_regras, cont);
	regra* regras = (regra*) calloc(cont, sizeof(regra));
	if(regras == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	cont = 0;
	
	for(i = 0; i < *quant_regras; i++){
		
		if(r[i].nula == 0){
			
			regras[cont] = r[i];
			cont++;
		}
	}
	
	*quant_regras = cont;
	
	return regras;
}

void setLBest(parametros param, regiao_pareto pareto, particula* enxame){
	
	int i;
	for(i = 0; i < param.quant_particulas; i++){
		
		if(dominaPorPareto( enxame[i].posicao, enxame[i].lBest, pareto ) == -1)
			enxame[i].lBest = enxame[i].posicao;
	}
}

int comparaSENS(const void *a, const void *b){
	regra *x = (regra *) a;
	regra *y = (regra *) b;
	
	if((*x).func_ob[SENS] > (*y).func_ob[SENS])
		return 1;
	if((*x).func_ob[SENS] < (*y).func_ob[SENS])
		return -1;
	return 0;
}

int comparaSPEC(const void *a, const void *b){
	regra *x = (regra *) a;
	regra *y = (regra *) b;
	
	if((*x).func_ob[SPEC] > (*y).func_ob[SPEC])
		return 1;
	if((*x).func_ob[SPEC] < (*y).func_ob[SPEC])
		return -1;
	return 0;
}

void atualizaCrowdingDistance(regiao_pareto pareto, int objetivo, regra* regras, int quant_regras){
	
	int i,j;
	/* imprimeDominioParetoComObjetivos(*pareto, quant_atrib, param);
	
	printf("\nAntigas Crowding Distances:\n");
			for(j = 0; j < (*pareto).quant_sol_pareto; j++){
		
				printf("%f ", (*pareto).solucoes[j].crowding_distance);
			} */
	
	switch(objetivo){
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
	
	for(i = 0; i < quant_regras; i++){
		if(regras[i].func_ob[objetivo] == regras[0].func_ob[objetivo] || regras[i].func_ob[objetivo] == regras[quant_regras - 1].func_ob[objetivo])
			regras[i].crowding_distance = INFINITO;
	}
	
	//regras[0].crowding_distance = INFINITO;
	for(i = 1; i < quant_regras - 1; i++){
		if(regras[i].crowding_distance != INFINITO)
			regras[i].crowding_distance += regras[i+1].func_ob[objetivo] - regras[i-1].func_ob[objetivo]; 
	}
	//regras[quant_regras - 1].crowding_distance = INFINITO;
	
	/* printf("\nNovas Crowding Distances:\n");
			for(j = 0; j < (*pareto).quant_sol_pareto; j++){
		
				printf("%f ", (*pareto).solucoes[j].crowding_distance);
			} */
}

void atualizaCrowdingDistances(regiao_pareto pareto, regra* regras, int quant_regras){
	
	int i;
	for(i = 0; i < quant_func_ob; i++){
		
		if(pareto.func_obj[i] == 1)
			atualizaCrowdingDistance(pareto, i, regras, quant_regras);
	}
	
}	

int comparaDominadores(const void *a, const void *b){
	regra *x = (regra *) a;
	regra *y = (regra *) b;
	
	if((*x).quant_dominadores > (*y).quant_dominadores)
		return 1;
	if((*x).quant_dominadores < (*y).quant_dominadores)
		return -1;
	return 0;
}

void ordenaRegrasMenosDominadas(regra* regras, int quant_regras){
	qsort(regras, quant_regras, sizeof(regra), comparaDominadores);
}

int comparaCrowdDistances(const void *a, const void *b){
	regra *x = (regra *) a;
	regra *y = (regra *) b;
	
	if((*x).crowding_distance > (*y).crowding_distance)
		return -1;
	if((*x).crowding_distance < (*y).crowding_distance)
		return 1;
	return 0;
}

void ordenaRegrasMenoresCrowdDistances(regra* regras, int quant_regras){
	qsort(regras, quant_regras, sizeof(regra), comparaCrowdDistances);
}
	
void setGBest(parametros param, regiao_pareto pareto, particula* enxame){
	
	int i;
	regra r1, r2;
	
	for(i = 0; i < param.quant_particulas; i++){
		
		r1 = pareto.solucoes[rand()%pareto.quant_sol_pareto];
		r2 = pareto.solucoes[rand()%pareto.quant_sol_pareto];
		
		if(r1.crowding_distance > r2.crowding_distance)
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
		phi = (rand()%101)/100.0;
	}while(phi > MAX_OMEGA);
	
	return phi;
}

double getConst(){
	
	return 2;//1.5 + (rand()%101)/100.0;
}

int posicaoMenosPosicao(int p1, int p2){
	//printf("\nposicao1 = %d", p1);
	//printf("\nposicao2 = %d", p2);
	if(p1 != p2)
		return p1;
	else
		return -2;	// -2 representa um valor fora do dom�nio dos atributo da regra;
}

double coefVezesPosicao(double c, int posicao){
	
	double phi = getPhi();
	//printf("\nposicao = %d", posicao);
	//printf("\nphi = %f", phi);
	//printf("\nc = %f", c);
	if(posicao == -2)
		return 0;
	else if (c*phi < 1)
		return c*phi;
	else
		return 1;
}

void coefVezesVelocidade(double* velocidade, int quant_vel){
	
	double coef = getOmega();
	int i;
	
	for(i = 0; i < quant_vel; i++){
		
		if(coef*velocidade[i] < 1)
			velocidade[i] = coef*velocidade[i];
		else
			velocidade[i] = 1;
	}
}

/*Dada a velocidade calculada de um valor de um atributo, verifica
se esse valor � maior que o correspondente dentro do vetor de velocidades*/
void somaVelocidades(double* velocidades, double vel, int atributo){
	//printf("\natributo: %d", atributo);
	//printf("\nvelocidades[] = %f, vel = %f", velocidades[atributo], vel);
	if(velocidades[atributo + 1] < vel)
		velocidades[atributo + 1] = vel;
}

void setVelocidadeParticula(particula* p, atributo* atrib, int quant_atrib, parametros param){
	
	double c1 = getConst();
	double c2 = getConst();
	
	int i;
	for(i = 0; i < quant_atrib; i++){
		
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
	
	for(i = 0; i < quant_vel; i++){
		
		if(velocidades[i] > vel)
			vel = velocidades[i];
	}
	
	return vel;
}

double getRandFloat(double limite){
	
	double r;
	
	do{
		r = (rand()%101)/100.0;
	}while(r > limite);
	
	return r;
}

int roleta (double* velocidade, atributo atrib){
	
	double limite = getMaiorVelocidade(velocidade, atrib.quant_real + 1);
	double alfa = getRandFloat(limite);
	int posicao;
	//printf("\nRoleta:");
	//printf("\nlimite = %f", limite);
	//printf("\nalfa = %f", alfa);
	
	do{
		posicao = rand()%( atrib.quant_real + 1) - 1;
		//printf("\nposicao = %d", posicao);
		//printf("\nvelocidade = %f\n", velocidade[posicao]);
	}while(velocidade[posicao + 1] < alfa);
	
	return posicao;	
}

void setPosicaoParticula(particula* p, atributo* atrib, int quant_atrib){
	
	int i;
	for(i = 0; i < quant_atrib; i++){
		
		(*p).posicao.valores[i] = roleta((*p).velocidade[i], atrib[i]);
	}
}

//cria o arquivo de sa�da contendo as solu��es do algoritmo (execucao - corresponde ao n�mero da execu��o do algoritmo como um todo)
FILE* criaArquivo(int execucao){
	
	char nome_aqr[22];
	snprintf(nome_aqr, sizeof(nome_aqr), "solucoes - %d.txt", execucao); 
	
	FILE* output = fopen(nome_aqr, "w");
	
	if (output == NULL){																											
		
		printf("Nao foi possivel criar arquivo!\n");
		exit(1);																										
	}
	else
		return output;	
}

void imprimeNomesRegraEmArquivo(regra r, atributo* atrib, int quant_atrib, FILE* output){
	
	int i;
	for(i = 0; i < quant_atrib - 1; i++){
		
		fprintf(output, "%s ", getNomeValorAtributo(atrib, quant_atrib, i, r.valores[i]));
	}
	
	fprintf(output, "-> %s\n", getNomeValorAtributo(atrib, quant_atrib, quant_atrib - 1, r.valores[quant_atrib - 1]));
}

void insereNomesSolucoesArquivo(regiao_pareto solucao, parametros param, FILE* output, atributo* atrib, int quant_atrib){
	
	int i;
	for(i = 0; i < solucao.quant_sol_pareto; i++){
		imprimeNomesRegraEmArquivo(solucao.solucoes[i], atrib, quant_atrib, output);
	}
}

void insereObjEmArquivo(regiao_pareto solucao, particula* enxame, parametros param, FILE* output, int interacao){
	
	int i,j;
	
	fprintf(output, "Intera��o: %d\n", interacao);
	
	fprintf(output, "Enxame:\n");
	for(i = 0; i < param.quant_particulas; i++){
		for(j = 0; j < quant_func_ob; j++){
				
			if(param.funcoes_obj_pareto[j] == '1'){				
				
				fprintf(output, "%f\t", enxame[i].posicao.func_ob[j]);
			}
		}
		fprintf(output, "\n");
	}	
	fprintf(output, "\n");
	
	fprintf(output, "Solu��es:\n");
	for(i = 0; i < solucao.quant_sol_pareto; i++){
		for(j = 0; j < quant_func_ob; j++){
				
			if(param.funcoes_obj_pareto[j] == '1'){				
				
				fprintf(output, "%f\t", solucao.solucoes[i].func_ob[j]);
			}
		}
		fprintf(output, "\n");
	}
	fprintf(output, "\n");
}

insereSolucoesEmArquivo(regiao_pareto solucao, parametros param, FILE* output){
	
	int i,j;
	
	fprintf(output, "Fim:\n\n");
	fprintf(output, "Solu��es:\n");
	for(i = 0; i < solucao.quant_sol_pareto; i++){
		for(j = 0; j < quant_func_ob; j++){
				
			if(param.funcoes_obj_pareto[j] == '1'){				
				
				fprintf(output, "%f\t", solucao.solucoes[i].func_ob[j]);
			}
		}
		fprintf(output, "\n");
	}
	fprintf(output, "\n");
}

void atualizaDominadores(regra* regras, int quant_regras, regiao_pareto pareto){
	
	zeraQuantDominadoresRegras(regras, quant_regras);
	int i,j;																										  
	
	for(i = 0; i < quant_regras - 1; i++){
		
		for(j = i+1; j < quant_regras; j++){
			
			if( dominaPorPareto(regras[i], regras[j], pareto) == -1 ){
				regras[j].quant_dominadores++;
			}
			else if( dominaPorPareto(regras[i], regras[j], pareto) == 1 ){
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
	while((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){
		
		if(regras[i].quant_dominadores == 0 && (regras[i].crowding_distance != INFINITO || cont != 1)){
			inserePareto(pareto, regras[i]);
			
			if(regras[i].crowding_distance == INFINITO)
				cont = 1;
		}
		i++;
	}
	
	ordenaRegrasMenosDominadas(regras, quant_regras);
	
	i = 0;
	while((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){
		
		if(regras[i].quant_dominadores != 0){
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
	while((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){
		
		if(regras[i].quant_dominadores == 0){
			inserePareto(pareto, regras[i]);
		}
		i++;
	}
	
	i = 0;
	
	while((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){
		
		if(regras[i].quant_dominadores != 0){
			inserePareto(pareto, regras[i]);
		}
		i++;
	}
}

void removeDominadas(regiao_pareto* pareto){
	
	int i;
	for(i = 0; i < (*pareto).quant_sol_pareto; i++){
		
		if( (*pareto).solucoes[i].quant_dominadores != 0 ){
			
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
	while((*pareto).quant_sol_pareto < param.tamanho_arquivo && i < quant_regras){
			
		if(regras[i].quant_dominadores == 0){
			inserePareto(pareto, regras[i]);
		}
		i++;
	}
}

regiao_pareto SPSO(parametros param, exemplo* exemplos, int quant_exemp, atributo* atrib, int quant_atrib, FILE* output){
	
	particula* enxame = criaEnxame(param, atrib, quant_atrib);
	calculaObjetivosEnxame(enxame, param, exemplos, quant_exemp, atrib, quant_atrib);
	regra* regras_enxame = enxameParaRegras(param, enxame);
	regiao_pareto pareto = inicializaPareto(param);
	
	if(param.metodo_dopagem_solucao == 0)																			// ----->> insere dominadas na solu��o (insereDominadasPorOrdemDeCrowdDistance ou insereDominadasPorOrdemDeDominadores())
		insereDominadasPorOrdemDeDominadores(param, regras_enxame, param.quant_particulas, &pareto);	
	else if(param.metodo_dopagem_solucao == 1)
		insereDominadasPorOrdemDeCrowdDistance(param, regras_enxame, param.quant_particulas, &pareto);
	else
		insereNaoDominadasPorOrdemDeCrowdDistance(param, regras_enxame, param.quant_particulas, &pareto);
	
	setGBest(param, pareto, enxame);
	
	regra* total;
	//FILE* arq_solucao = criaArquivo();
	//insereObjEmArquivo(pareto, enxame, param, arq_solucao, 0);
	
	int i,j, quant_regras;
	for(i = 0; i < param.bl_interacoes; i++){
		for(j = 0; j < param.quant_particulas; j++){
			
			if(param.classe == -1){
				
				setVelocidadeParticula(&enxame[j], atrib, quant_atrib, param);
				setPosicaoParticula(&enxame[j], atrib, quant_atrib);
			}
			else{
				
				setVelocidadeParticula(&enxame[j], atrib, quant_atrib - 1, param);
				setPosicaoParticula(&enxame[j], atrib, quant_atrib - 1);
			}
		}
		
		calculaObjetivosEnxame(enxame, param, exemplos, quant_exemp, atrib, quant_atrib);
		setLBest(param, pareto, enxame);
		regras_enxame = enxameParaRegras(param, enxame);
		total = uneRegras(regras_enxame, param.quant_particulas, pareto.solucoes, pareto.quant_sol_pareto);
		quant_regras = param.quant_particulas + pareto.quant_sol_pareto;
		total = apagaRegrasIguais(total, &quant_regras, quant_atrib);
		pareto = inicializaPareto(param);
		
		if(param.metodo_dopagem_solucao == 0)																		// ----->> insere dominadas na solu��o (insereDominadasPorOrdemDeCrowdDistance ou insereDominadasPorOrdemDeDominadores())
			insereDominadasPorOrdemDeDominadores(param, total, quant_regras, &pareto);	
		else if(param.metodo_dopagem_solucao == 1)
			insereDominadasPorOrdemDeCrowdDistance(param, total, quant_regras, &pareto);
		else
			insereNaoDominadasPorOrdemDeCrowdDistance(param, total, quant_regras, &pareto);
		
		setGBest(param, pareto, enxame);
		insereObjEmArquivo(pareto, enxame, param, output, i + 1);
	}
	removeDominadas(&pareto);
	insereSolucoesEmArquivo(pareto, param, output);
	insereNomesSolucoesArquivo(pareto, param, output, atrib, quant_atrib);
	imprimeDominioPareto(pareto, quant_atrib);
	
	free(enxame);
	free(regras_enxame);
	free(total);
	fclose(output);
	
	return pareto;
}

/* regiao_pareto SPSO(parametros param, exemplo* exemplos, int quant_exemp, atributo* atrib, int quant_atrib){
	
	particula* enxame = criaEnxame(param, atrib, quant_atrib);
	calculaObjetivosEnxame(enxame, param, exemplos, quant_exemp, atrib, quant_atrib);
	
	//imprimeEnxame(enxame, param, atrib, quant_atrib);
	
	regra* regras_enxame = enxameParaRegras(param, enxame);
	regiao_pareto pareto = dominanciaDePareto(param, regras_enxame, param.quant_particulas);
	atualizaCrowdingDistances(&pareto);
	
	int k;
	printf("\nEnxame:");
	for(k = 0; k < param.quant_particulas; k++){
		printf("CDs: %f\n", regras_enxame[k].crowding_distance);
	}
	printf("\nPareto antes:");
	for(k = 0; k < pareto.quant_sol_pareto; k++){
		printf("CDs: %f\n", pareto.solucoes[k].crowding_distance);
	}
	
	//ordenaRegrasMenoresCrowdDistances(regras_enxame, param.quant_particulas);
	
	ordenaRegrasMenosDominadas(regras_enxame, param.quant_particulas);
	insereDominadasPorOrdemDeDominadores(param, regras_enxame, param.quant_particulas, &pareto);
	
	printf("\nPareto depois:");
	for(k = 0; k < pareto.quant_sol_pareto; k++){
		printf("CDs: %d\n", pareto.solucoes[k].crowding_distance);
	}
		
	int k;
	printf("\n");
	for(k = 0; k < param.quant_particulas; k++){
		printf("Quantidade de dominadores: %d\n", regras_enxame[k].quant_dominadores);
	}
	ordenaRegrasMenosDominadas(regras_enxame, param.quant_particulas);
	printf("\ndepois:\n");
	for(k = 0; k < param.quant_particulas; k++){
		printf("Quantidade de dominadores: %d\n", regras_enxame[k].quant_dominadores);
	}
	
	//imprimeDominioPareto(pareto, quant_atrib);
	
	setGBest(param, pareto, enxame);
	
	//imprimeEnxame(enxame, param, atrib, quant_atrib);
	
	regra* total;
	
	//imprimeDominioPareto(pareto, quant_atrib);
	
	FILE* arq_solucao = criaArquivo();
	insereObjEmArquivo(pareto, enxame, param, arq_solucao, 0);
	
	int i,j, quant_regras;
	for(i = 0; i < param.bl_interacoes; i++){
		
		for(j = 0; j < param.quant_particulas; j++){
			
			if(param.classe == -1){
				
				setVelocidadeParticula(&enxame[j], atrib, quant_atrib);
				setPosicaoParticula(&enxame[j], atrib, quant_atrib);
			}
			else{
				
				setVelocidadeParticula(&enxame[j], atrib, quant_atrib - 1);
				setPosicaoParticula(&enxame[j], atrib, quant_atrib - 1);
			}
		}
		
		calculaObjetivosEnxame(enxame, param, exemplos, quant_exemp, atrib, quant_atrib);
		//printf("\nInicio Enxame");
		//imprimeEnxame(enxame, param, atrib, quant_atrib);
		//printf("\nFim Enxame");
		setLBest(param, pareto, enxame);
		
		//printf("\nInicio Enxame1");
		//imprimeEnxame(enxame, param, atrib, quant_atrib);
		//printf("\nFim Enxame1");
		
		regras_enxame = enxameParaRegras(param, enxame);
		total = uneRegras(regras_enxame, param.quant_particulas, pareto.solucoes, pareto.quant_sol_pareto);
		quant_regras = param.quant_particulas + pareto.quant_sol_pareto;
		
		printf("\nantes:\n");
		int k;
		for(k = 0; k < quant_regras; k++){
			imprimeRegra(total[k], quant_atrib);
		}
		total = apagaRegrasIguais(total, &quant_regras, quant_atrib);
		
		int k;
		printf("\n");
		for(k = 0; k < quant_regras; k++){
			imprimeRegra(total[k], quant_atrib);
			printf("Quantidade de dominadores: %d\n", total[k].quant_dominadores);
		}
		
		pareto = dominanciaDePareto(param, total, quant_regras);
		atualizaCrowdingDistances(&pareto);
		
		ordenaRegrasMenosDominadas(total, quant_regras);
		insereDominadasPorOrdemDeDominadores(param, total, quant_regras, &pareto);
		
		setGBest(param, pareto, enxame);
		//printf("\nok");
		//printf("\n\nAQUI\n\n");
		//imprimeDominioPareto(pareto, quant_atrib);
		insereObjEmArquivo(pareto, enxame, param, arq_solucao, i + 1);
	}
	removeDominadas(&pareto);
	//imprimeDominioParetoComObjetivos(pareto, quant_atrib, param);
	
	imprimeDominioPareto(pareto, quant_atrib);
	
	free(enxame);
	free(regras_enxame);
	free(total);
	fclose(arq_solucao);
	
	return pareto;
} */

//////////////////////////////////////// FIM S-PSO ////////////////////////////////////////

void criaArquivoObjSolucao(regiao_pareto solucao, int indice_obj, int indice_arq){
	
	char nome_aqr[22];
	snprintf(nome_aqr, sizeof(nome_aqr), "objetivo(%d).txt",indice_arq); 
	
	FILE* output = fopen(nome_aqr, "w");
	
	if (output == NULL){																											
		
		printf("Nao foi possivel criar arquivo!\n");																					
		printf("\n%s", nome_aqr);
		printf("\nlenght = %d", strlen(nome_aqr));
		exit(1);																										
	}
	else{
		fprintf(output, "%s\n", nome_aqr);
		
		int i;
		for(i = 0; i < solucao.quant_sol_pareto; i++){
			
			fprintf(output, "%f\n", solucao.solucoes[i].func_ob[indice_obj]);
		}					
		
		fclose(output);
	}
}

void criaArquivosObjSolucao(regiao_pareto solucao, parametros param){
	
	int cont = 1;
	int i,j;
	
	for(i = 0; i < quant_func_ob; i++){
			
			if(param.funcoes_obj_pareto[i] == '1'){				
				
				criaArquivoObjSolucao(solucao, i, cont);
				cont++;
			}
		}
}

void imprimeSolucao(regiao_pareto solucao, atributo* atrib, int quant_atrib){
	
	printf("\n#SOLU��ES SPSO:\n");
	
	int i;
	for(i = 0; i < solucao.quant_sol_pareto; i++){
		
		imprimeNomesRegra(solucao.solucoes[i], atrib, quant_atrib);
	}
}

regra mapeiaLinhaWeka(atributo* atributos, int quant_atrib, char* linha){
	
	regra r = inicializaRegra();
	
	int i;
	for(i = 0; i < quant_atrib; i++)																							// preenche regra com valores vazios;
		r.valores[i] = -1;
	
	int indice_atributo = 0;																									// representa o �ndice do atributo atual da linha sendo analisado;
	int indice_letra = 0;																										// representa a letra atual da linha sendo analisada;	
	int tamanho_aux = 200;
	char aux[tamanho_aux];																										// auxiliar que receber� as palavras analisadas da linha;
	int indice_aux = 0;																											// �ndice da letra atual analisada de aux;
	char* str_aux;																												// string din�mica que receber� o valor de aux;
	int atributo = -1;																											// �ndice do atributo atual sendo analisado;																												
	int j;
	int flag = 0;																												// indica se todos os atributos antes da classe j� foram verificados;
	
	//Trata dos atributos anteriores � classe
	
	while(linha[indice_letra] < 48 || linha[indice_letra] > 57){

		while(linha[indice_letra] != '='){
			aux[indice_aux] = linha[indice_letra];
			indice_aux++;
			indice_letra++;
		}
		
		str_aux = (char*) calloc(indice_aux + 1, sizeof(char));
		strncpy(str_aux, aux, indice_aux);
		str_aux[indice_aux] = '\0';
		
		for(j = 0; j < quant_atrib; j++){
			if(strcmp(atributos[j].nome, str_aux) == 0 ){
				atributo = j;
				//printf("\nAqui1:%s - indice = %d", atributos[j].nome, j);
			}
		}
		
		//printf("\tAqui2:%s - t = %d", str_aux, strlen(str_aux));
		indice_aux = 0;
		linha = linha + 1;																											// pula s�mbolo "="
		free(str_aux);																												// reseta string auxiliar;
		
		while(linha[indice_letra] != ' '){
			aux[indice_aux] = linha[indice_letra];
			indice_aux++;
			indice_letra++;
		}
		
		str_aux = (char*) calloc(indice_aux + 1, sizeof(char));
		strncpy(str_aux, aux, indice_aux);
		str_aux[indice_aux] = '\0';
		
		for(j = 0; j < atributos[atributo].quant_real; j++){
			if(strcmp(atributos[atributo].valor[j], str_aux) == 0 ){
				r.valores[atributo] = j;
				//printf("\tAqui2:%s - t = %d, j = %d", atributos[atributo].valor[j], strlen(atributos[atributo].valor[j]), j);
			}
		}
		
		//printf("\tAqui4:%s - t = %d", str_aux, strlen(str_aux));
		indice_aux = 0;
		linha = linha + 1;																											// pula s�mbolo " "
		free(str_aux);
	}
	
	//Trata atributo da classe
	
	while(linha[0] != '>'){
		linha = linha + 1;
	}
	
	linha = linha + 2;																												// pula s�mbolos "> "
	indice_letra = 0;
	
	while(linha[indice_letra] != '='){
		aux[indice_aux] = linha[indice_letra];
		indice_aux++;
		indice_letra++;
	}
	
	str_aux = (char*) calloc(indice_aux + 1, sizeof(char));
	strncpy(str_aux, aux, indice_aux);
	str_aux[indice_aux] = '\0';
	
	if(strcmp(atributos[quant_atrib - 1].nome, str_aux) == 0 ){
		atributo = quant_atrib - 1;
		//printf("\nAqui3:%s - t = %d", atributos[quant_atrib - 1].nome, strlen(atributos[quant_atrib - 1].nome));
	}
	
	
	//printf("\tAqui2:%s - t = %d", str_aux, strlen(str_aux));
	indice_aux = 0;
	linha = linha + 1;																											// pula s�mbolo "="
	free(str_aux);																												// reseta string auxiliar;
	
	while((linha[indice_letra] < 48 || linha[indice_letra] > 57) && linha[indice_letra] != ' '){
		aux[indice_aux] = linha[indice_letra];
		indice_aux++;
		indice_letra++;
	}
	
	str_aux = (char*) calloc(indice_aux + 1, sizeof(char));
	strncpy(str_aux, aux, indice_aux);
	str_aux[indice_aux] = '\0';
	
	for(j = 0; j < atributos[atributo].quant_real; j++){
		//printf("\tAqui4:%s - t = %d, j = %d, str_aux = %s, strlen(str_aux) = %d", atributos[atributo].valor[j], strlen(atributos[atributo].valor[j]), j, str_aux, strlen(str_aux));
		if(strcmp(atributos[atributo].valor[j], str_aux) == 0 ){
			r.valores[atributo] = j;
			//printf("\tAqui5:%s - t = %d, j = %d", atributos[atributo].valor[j], strlen(atributos[atributo].valor[j]), j);
		}
	}
	
	//printf("\tAqui4:%s - t = %d", str_aux, strlen(str_aux));
	
	r.nula = 0;
	return r;
}

/*Converte regras de um arquivo Weka num vetor de regras mapeadas*/
regra* getRegrasWeka(atributo* atributos, int quant_atrib, int* quant_regras){
	
	FILE* weka = fopen("regras_weka.txt", "r");
	if (weka == NULL){																											// caso o  arquivo n�o exista
		
		printf("Arquivo Weka n�o encontrado!\n");																				// emite mensagem de erro
		exit(1);																												// e finaliza;
	}
	
	char* pont_aux;
	*quant_regras = 0;
	
	while(!feof(weka)){																											// conta quantidade de regras no arquivo Weka;
		
		pont_aux = localizaString(". ", weka);
		//printf("\npont_aux = %s",pont_aux);
		if(pont_aux != NULL)
			(*quant_regras)++;
	}
	rewind(weka);
	//printf("\nquant_regras = %d", *quant_regras);
	regra* regras = (regra*) calloc(*quant_regras, sizeof(regra));
	
	const int tam_str_aux = 1000;
	int indice = 0;
	while(!feof(weka)){
		char str_aux [tam_str_aux];																								// string auxiliar que receber� a linha do arquivo Weka a ser lida;
		fgets(str_aux, tam_str_aux, weka);	
		
		char* linha = strstr(str_aux, ". ");
		if(linha != NULL ){
			linha = linha + 2;
			//printf("\nAqui:%s", linha);
			regras[indice] = mapeiaLinhaWeka(atributos, quant_atrib, linha);
			indice++;
		};
	}
	
	printf("\nRegras Weka:\n");
	int i;
	for(i = 0; i < indice; i++){
		imprimeRegra(regras[i], quant_atrib);
	}
	
	return regras;
}

insereSolucoesWekaEmArquivo(regra* regras, int quant_regras, parametros param, FILE* output){
	
	int i,j;
	
	fprintf(output, "Solu��es:\n");
	for(i = 0; i < quant_regras; i++){
		for(j = 0; j < quant_func_ob; j++){
				
			if(param.funcoes_obj_pareto[j] == '1'){
				fprintf(output, "%f\t", regras[i].func_ob[j]);
			}
		}
		fprintf(output, "\n");
	}
	fprintf(output, "\n");
	fclose(output);
}

void calculaObjetivosRegrasWeka(atributo* atrib, int quant_atrib, exemplo* exemplos, int quant_exemp, parametros param){
	
	int quant_regras;
	regra* regras = getRegrasWeka(atrib, quant_atrib, &quant_regras);
	
	int i;
	for(i = 0; i < quant_regras; i++){
		preencheMatrizCont(&regras[i],exemplos, quant_exemp, quant_atrib );
		calculaFuncoesObj(&regras[i],quant_atrib, atrib, quant_exemp );
		//imprimeMatrizCont(regras[i]);
		//imprimeFuncoesObj(regras[i]);
		//printf("\n%f", regras[i].func_ob[4]);
	}
	
	FILE* arq_solucao_weka = fopen("solu��es_weka.txt", "w");
	insereSolucoesWekaEmArquivo(regras, quant_regras, param, arq_solucao_weka);
}

int main(/*int argc, char* argv[]*/){

	system("cls");
	//printf("ARG1 = %s\n", argv[1]);
	
	//VARI�VEIS
	FILE* arq_parametros = carregarArq("parametros.txt");
	parametros param;																											// par�metros para o algoritmo de busca local;
	carregaParametros(arq_parametros, &param);
	FILE* input = carregarArq(param.arquivo);
	int quant_atrib = contaAtributos(input);																					// quantidade de atributos dos elementos da base de dados;
	atributo* atrib = (atributo*) calloc(quant_atrib, sizeof(atributo));														// ponteiro para aloca��o din�mica do vetor com atributos;
	if(atrib == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	int quant_exemp = contaExemplos(input);																						// quantidade de exemplos da base de dados;
	exemplo* exemplos = (exemplo*) calloc(quant_exemp, sizeof(exemplo));														// ponteiro para aloca��o din�mica do vetor com exemplos;
	if(exemplos == NULL){
		
		printf("\nErro de alcocacao!");
		exit(1);
	}
	
	//INICIA M�DULOS
	
	srand( (unsigned) time(NULL) );
	imprimeParametros(param);
	processaAtributos(atrib, quant_atrib, input);																				//preenche vetor de atributos da classe;
	imprimeAtributos(atrib, quant_atrib);																						//imprime todas as informa��es contidas no vetor de atributos;
	processaExemplos(atrib, quant_atrib, exemplos,input);
	//calculaObjetivosRegrasWeka(atrib, quant_atrib, exemplos, quant_exemp, param);
	
	//imprimeExemplosInt(exemplos, quant_exemp);
	//regra r = geraRegra(param, atrib, quant_atrib, param.classe);
	//imprimeRegra(r, quant_atrib);
	//preencheMatrizCont(&r, exemplos, quant_exemp, quant_atrib);
	//imprimeMatrizCont(r);
	//calculaFuncoesObj(&r, quant_atrib, atrib, quant_exemp);
	//imprimeFuncoesObj(r);
	
	//buscaLocal(param, atrib, quant_atrib, exemplos, quant_exemp);
	//regra* regras = geraRegras(param, param.quant_regras_pareto, atrib, quant_atrib, param.classe, exemplos, quant_exemp);
	//regiao_pareto pareto = dominanciaDePareto(param, regras, param.quant_regras_pareto);
	//imprimeDominioPareto(pareto, quant_atrib);
	
	int i;
	for(i = 0; i < param.execucoes; i++){
		FILE* arq_solucao = criaArquivo(i);
		regiao_pareto solucao = SPSO(param, exemplos, quant_exemp, atrib, quant_atrib, arq_solucao);
	}
	
	//imprimeSolucao(solucao, atrib, quant_atrib);
	//criaArquivosObjSolucao(solucao, param);
	
	fclose(arq_parametros);
	fclose(input);
	
	system("pause");
	
	return 0;
}
