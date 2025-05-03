/-- A first-order language consists of a type of functions of every natural-number arity and a
  type of relations of every natural-number arity. -/
@[nolint checkUnivs]
structure Language where
  /-- For every arity, a `Type*` of functions of that arity -/
  Functions : ℕ → Type u
  /-- For every arity, a `Type*` of relations of that arity -/
  Relations : ℕ → Type v


/-- A language is relational when it has no function symbols. -/
abbrev IsRelational : Prop := ∀ n, IsEmpty (L.Functions n)


/-- A language is algebraic when it has no relation symbols. -/
abbrev IsAlgebraic : Prop := ∀ n, IsEmpty (L.Relations n)


/-- The empty language has no symbols. -/
protected def empty : Language :=
  ⟨fun _ => Empty, fun _ => Empty⟩
  deriving IsAlgebraic, IsRelational


instance : Inhabited Language :=
  ⟨Language.empty⟩


/-- The sum of two languages consists of the disjoint union of their symbols. -/
protected def sum (L' : Language.{u', v'}) : Language :=
  ⟨fun n => L.Functions n ⊕ L'.Functions n, fun n => L.Relations n ⊕ L'.Relations n⟩


/-- The type of constants in a given language. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
protected abbrev Constants :=
  L.Functions 0


/-- The type of symbols in a given language. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
abbrev Symbols :=
  (Σ l, L.Functions l) ⊕ (Σ l, L.Relations l)


/-- The cardinality of a language is the cardinality of its type of symbols. -/
def card : Cardinal :=
  #L.Symbols


theorem card_eq_card_functions_add_card_relations :
    L.card =
      (Cardinal.sum fun l => Cardinal.lift.{v} #(L.Functions l)) +
        Cardinal.sum fun l => Cardinal.lift.{u} #(L.Relations l) := by
  /-
    L : FirstOrder.Language
    ⊢ Eq L.card (HAdd.hAdd (Cardinal.sum fun l => Cardinal.lift.{v, u} (Cardinal.m …
  -/
  simp only [card, mk_sum, mk_sigma, lift_sum]
  /-
    🎉 no goals
  -/


instance isRelational_sum [L.IsRelational] [L'.IsRelational] : IsRelational (L.sum L') :=
  fun _ => instIsEmptySum


instance isAlgebraic_sum [L.IsAlgebraic] [L'.IsAlgebraic] : IsAlgebraic (L.sum L') :=
  fun _ => instIsEmptySum


@[simp]
theorem empty_card : Language.empty.card = 0 := by simp only [card, mk_sum, mk_sigma, mk_eq_zero,
  sum_const, mk_eq_aleph0, lift_id', mul_zero, add_zero]


instance isEmpty_empty : IsEmpty Language.empty.Symbols := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    ⊢ IsEmpty FirstOrder.Language.empty.Symbols
  -/
  simp only [Language.Symbols, isEmpty_sum, isEmpty_sigma]
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    ⊢ And (∀ (a : Nat), IsEmpty (FirstOrder.Language.empty.Functions a)) (∀ (a : N …
  -/
  exact ⟨fun _ => inferInstance, fun _ => inferInstance⟩
  /-
    🎉 no goals
  -/


instance Countable.countable_functions [h : Countable L.Symbols] : Countable (Σl, L.Functions l) :=
  @Function.Injective.countable _ _ h _ Sum.inl_injective


@[simp]
theorem card_functions_sum (i : ℕ) :
    #((L.sum L').Functions i)
      = (Cardinal.lift.{u'} #(L.Functions i) + Cardinal.lift.{u} #(L'.Functions i) : Cardinal) := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    i : Nat
    ⊢ Eq (Cardinal.mk ((L.sum L').Functions i)) (HAdd.hAdd (Cardinal.lift.{u', u}  …
  -/
  simp [Language.sum]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_relations_sum (i : ℕ) :
    #((L.sum L').Relations i) =
      Cardinal.lift.{v'} #(L.Relations i) + Cardinal.lift.{v} #(L'.Relations i) := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    i : Nat
    ⊢ Eq (Cardinal.mk ((L.sum L').Relations i)) (HAdd.hAdd (Cardinal.lift.{v', v}  …
  -/
  simp [Language.sum]
  /-
    🎉 no goals
  -/


theorem card_sum :
    (L.sum L').card = Cardinal.lift.{max u' v'} L.card + Cardinal.lift.{max u v} L'.card := by
  simp only [card, mk_sum, mk_sigma, card_functions_sum, sum_add_distrib', lift_add, lift_sum,
    lift_lift, card_relations_sum, add_assoc,
    add_comm (Cardinal.sum fun i => (#(L'.Functions i)).lift)]


/-- Passes a `DecidableEq` instance on a type of function symbols through the  `Language`
constructor. Despite the fact that this is proven by `inferInstance`, it is still needed -
see the `example`s in `ModelTheory/Ring/Basic`. -/
instance instDecidableEqFunctions {f : ℕ → Type*} {R : ℕ → Type*} (n : ℕ) [DecidableEq (f n)] :
    DecidableEq ((⟨f, R⟩ : Language).Functions n) := inferInstance


/-- Passes a `DecidableEq` instance on a type of relation symbols through the  `Language`
constructor. Despite the fact that this is proven by `inferInstance`, it is still needed -
see the `example`s in `ModelTheory/Ring/Basic`. -/
instance instDecidableEqRelations {f : ℕ → Type*} {R : ℕ → Type*} (n : ℕ) [DecidableEq (R n)] :
    DecidableEq ((⟨f, R⟩ : Language).Relations n) := inferInstance


/-- A first-order structure on a type `M` consists of interpretations of all the symbols in a given
  language. Each function of arity `n` is interpreted as a function sending tuples of length `n`
  (modeled as `(Fin n → M)`) to `M`, and a relation of arity `n` is a function from tuples of length
  `n` to `Prop`. -/
@[ext]
class Structure where
  /-- Interpretation of the function symbols -/
  funMap : ∀ {n}, L.Functions n → (Fin n → M) → M := by
    exact fun {n} => isEmptyElim
  /-- Interpretation of the relation symbols -/
  RelMap : ∀ {n}, L.Relations n → (Fin n → M) → Prop := by
    exact fun {n} => isEmptyElim


/-- Used for defining `FirstOrder.Language.Theory.ModelType.instInhabited`. -/
def Inhabited.trivialStructure {α : Type*} [Inhabited α] : L.Structure α :=
  ⟨default, default⟩


/-- A homomorphism between first-order structures is a function that commutes with the
  interpretations of functions and maps tuples in one structure where a given relation is true to
  tuples in the second structure where that relation is still true. -/
structure Hom where
  /-- The underlying function of a homomorphism of structures -/
  toFun : M → N
  /-- The homomorphism commutes with the interpretations of the function symbols -/
  -- Porting note:
  -- The autoparam here used to be `obviously`. We would like to replace it with `aesop`
  -- but that isn't currently sufficient.
  -- See https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/Aesop.20and.20cases
  -- If that can be improved, we should change this to `by aesop` and remove the proofs below.
  map_fun' : ∀ {n} (f : L.Functions n) (x), toFun (funMap f x) = funMap f (toFun ∘ x) := by
    intros; trivial
  /-- The homomorphism sends related elements to related elements -/
  map_rel' : ∀ {n} (r : L.Relations n) (x), RelMap r x → RelMap r (toFun ∘ x) := by
    -- Porting note: see porting note on `Hom.map_fun'`
    intros; trivial


@[inherit_doc]
scoped[FirstOrder] notation:25 A " →[" L "] " B => FirstOrder.Language.Hom L A B


/-- An embedding of first-order structures is an embedding that commutes with the
  interpretations of functions and relations. -/
structure Embedding extends M ↪ N where
  map_fun' : ∀ {n} (f : L.Functions n) (x), toFun (funMap f x) = funMap f (toFun ∘ x) := by
    -- Porting note: see porting note on `Hom.map_fun'`
    intros; trivial
  map_rel' : ∀ {n} (r : L.Relations n) (x), RelMap r (toFun ∘ x) ↔ RelMap r x := by
    -- Porting note: see porting note on `Hom.map_fun'`
    intros; trivial


@[inherit_doc]
scoped[FirstOrder] notation:25 A " ↪[" L "] " B => FirstOrder.Language.Embedding L A B


/-- An equivalence of first-order structures is an equivalence that commutes with the
  interpretations of functions and relations. -/
structure Equiv extends M ≃ N where
  map_fun' : ∀ {n} (f : L.Functions n) (x), toFun (funMap f x) = funMap f (toFun ∘ x) := by
    -- Porting note: see porting note on `Hom.map_fun'`
    intros; trivial
  map_rel' : ∀ {n} (r : L.Relations n) (x), RelMap r (toFun ∘ x) ↔ RelMap r x := by
    -- Porting note: see porting note on `Hom.map_fun'`
    intros; trivial


@[inherit_doc]
scoped[FirstOrder] notation:25 A " ≃[" L "] " B => FirstOrder.Language.Equiv L A B

-- Porting note: was [L.Structure P] and [L.Structure Q]
-- The former reported an error.

/-- Interpretation of a constant symbol -/
@[coe]
def constantMap (c : L.Constants) : M := funMap c default


instance : CoeTC L.Constants M :=
  ⟨constantMap⟩


theorem funMap_eq_coe_constants {c : L.Constants} {x : Fin 0 → M} : funMap c x = c :=
  congr rfl (funext finZeroElim)


/-- Given a language with a nonempty type of constants, any structure will be nonempty. This cannot
  be a global instance, because `L` becomes a metavariable. -/
theorem nonempty_of_nonempty_constants [h : Nonempty L.Constants] : Nonempty M :=
  h.map (↑)


/-- `HomClass L F M N` states that `F` is a type of `L`-homomorphisms. You should extend this
  typeclass when you extend `FirstOrder.Language.Hom`. -/
class HomClass (L : outParam Language) (F : Type*) (M N : outParam Type*)
  [FunLike F M N] [L.Structure M] [L.Structure N] : Prop where
  map_fun : ∀ (φ : F) {n} (f : L.Functions n) (x), φ (funMap f x) = funMap f (φ ∘ x)
  map_rel : ∀ (φ : F) {n} (r : L.Relations n) (x), RelMap r x → RelMap r (φ ∘ x)


/-- `StrongHomClass L F M N` states that `F` is a type of `L`-homomorphisms which preserve
  relations in both directions. -/
class StrongHomClass (L : outParam Language) (F : Type*) (M N : outParam Type*)
  [FunLike F M N] [L.Structure M] [L.Structure N] : Prop where
  map_fun : ∀ (φ : F) {n} (f : L.Functions n) (x), φ (funMap f x) = funMap f (φ ∘ x)
  map_rel : ∀ (φ : F) {n} (r : L.Relations n) (x), RelMap r (φ ∘ x) ↔ RelMap r x

-- Porting note: using implicit brackets for `Structure` arguments

instance (priority := 100) StrongHomClass.homClass {F : Type*} [L.Structure M]
    [L.Structure N] [FunLike F M N] [StrongHomClass L F M N] : HomClass L F M N where
  map_fun := StrongHomClass.map_fun
  map_rel φ _ R x := (StrongHomClass.map_rel φ R x).2


/-- Not an instance to avoid a loop. -/
theorem HomClass.strongHomClassOfIsAlgebraic [L.IsAlgebraic] {F M N} [L.Structure M] [L.Structure N]
    [FunLike F M N] [HomClass L F M N] : StrongHomClass L F M N where
  map_fun := HomClass.map_fun
  map_rel _ _ := isEmptyElim


theorem HomClass.map_constants {F M N} [L.Structure M] [L.Structure N] [FunLike F M N]
    [HomClass L F M N] (φ : F) (c : L.Constants) : φ c = c :=
  (HomClass.map_fun φ c default).trans (congr rfl (funext default))


instance instFunLike : FunLike (M →[L] N) M N where
  coe := Hom.toFun
                             /-
                               L : FirstOrder.Language
                               L' : FirstOrder.Language
                               M : Type w
                               N : Type w'
                               inst✝³ : L.Structure M
                               inst✝² : L.Structure N
                               P : Type u_1
                               inst✝¹ : L.Structure P
                               Q : Type u_2
                               inst✝ : L.Structure Q
                               f g : L.Hom M N
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; cases h; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


instance homClass : HomClass L (M →[L] N) M N where
  map_fun := map_fun'
  map_rel := map_rel'


instance [L.IsAlgebraic] : StrongHomClass L (M →[L] N) M N :=
  HomClass.strongHomClassOfIsAlgebraic


@[simp]
theorem toFun_eq_coe {f : M →[L] N} : f.toFun = (f : M → N) :=
  rfl


@[ext]
theorem ext ⦃f g : M →[L] N⦄ (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


@[simp]
theorem map_fun (φ : M →[L] N) {n : ℕ} (f : L.Functions n) (x : Fin n → M) :
    φ (funMap f x) = funMap f (φ ∘ x) :=
  HomClass.map_fun φ f x


@[simp]
theorem map_constants (φ : M →[L] N) (c : L.Constants) : φ c = c :=
  HomClass.map_constants φ c


@[simp]
theorem map_rel (φ : M →[L] N) {n : ℕ} (r : L.Relations n) (x : Fin n → M) :
    RelMap r x → RelMap r (φ ∘ x) :=
  HomClass.map_rel φ r x


/-- The identity map from a structure to itself. -/
@[refl]
def id : M →[L] M where
  toFun m := m


instance : Inhabited (M →[L] M) :=
  ⟨id L M⟩


@[simp]
theorem id_apply (x : M) : id L M x = x :=
  rfl


/-- Composition of first-order homomorphisms. -/
@[trans]
def comp (hnp : N →[L] P) (hmn : M →[L] N) : M →[L] P where
  toFun := hnp ∘ hmn
  -- Porting note: should be done by autoparam?
                     /-
                       L : FirstOrder.Language
                       L' : FirstOrder.Language
                       M : Type w
                       N : Type w'
                       inst✝³ : L.Structure M
                       inst✝² : L.Structure N
                       P : Type u_1
                       inst✝¹ : L.Structure P
                       Q : Type u_2
                       inst✝ : L.Structure Q
                       hnp : L.Hom N P
                       hmn : L.Hom M N
                       n✝ : Nat
                       x✝¹ : L.Functions n✝
                       x✝ : Fin n✝ → M
                       ⊢ Eq (Function.comp (⇑hnp) (⇑hmn) (FirstOrder.Language.Structure.funMap x✝¹ x✝ …
                     -/
  map_fun' _ _ := by simp; rfl
                           /-
                             🎉 no goals
                           -/
  -- Porting note: should be done by autoparam?
  map_rel' _ _ h := map_rel _ _ _ (map_rel _ _ _ h)


@[simp]
theorem comp_apply (g : N →[L] P) (f : M →[L] N) (x : M) : g.comp f x = g (f x) :=
  rfl


/-- Composition of first-order homomorphisms is associative. -/
theorem comp_assoc (f : M →[L] N) (g : N →[L] P) (h : P →[L] Q) :
    (h.comp g).comp f = h.comp (g.comp f) :=
  rfl


@[simp]
theorem comp_id (f : M →[L] N) : f.comp (id L M) = f :=
  rfl


@[simp]
theorem id_comp (f : M →[L] N) : (id L N).comp f = f :=
  rfl


/-- Any element of a `HomClass` can be realized as a first_order homomorphism. -/
@[simps] def HomClass.toHom {F M N} [L.Structure M] [L.Structure N] [FunLike F M N]
    [HomClass L F M N] : F → M →[L] N := fun φ =>
  ⟨φ, HomClass.map_fun φ, HomClass.map_rel φ⟩


instance funLike : FunLike (M ↪[L] N) M N where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      P : Type u_1
      inst✝¹ : L.Structure P
      Q : Type u_2
      inst✝ : L.Structure Q
      f g : L.Embedding M N
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      P : Type u_1
      inst✝¹ : L.Structure P
      Q : Type u_2
      inst✝ : L.Structure Q
      g : L.Embedding M N
      toEmbedding✝ : Function.Embedding M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝. …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h : Eq ((fun f => f.toFun) { toEmbedding := toEmbedding✝, map_fun' := map_fun' …
      ⊢ Eq { toEmbedding := toEmbedding✝, map_fun' := map_fun'✝, map_rel' := map_rel …
    -/
    cases g
    /-
      case mk.mk
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      P : Type u_1
      inst✝¹ : L.Structure P
      Q : Type u_2
      inst✝ : L.Structure Q
      toEmbedding✝¹ : Function.Embedding M N
      map_fun'✝¹ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝ …
      map_rel'✝¹ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder. …
      toEmbedding✝ : Function.Embedding M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝. …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h : Eq ((fun f => f.toFun) { toEmbedding := toEmbedding✝¹, map_fun' := map_fun …
      ⊢ Eq { toEmbedding := toEmbedding✝¹, map_fun' := map_fun'✝¹, map_rel' := map_r …
    -/
    congr
    /-
      case mk.mk.e_toEmbedding
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      P : Type u_1
      inst✝¹ : L.Structure P
      Q : Type u_2
      inst✝ : L.Structure Q
      toEmbedding✝¹ : Function.Embedding M N
      map_fun'✝¹ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝ …
      map_rel'✝¹ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder. …
      toEmbedding✝ : Function.Embedding M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝. …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h : Eq ((fun f => f.toFun) { toEmbedding := toEmbedding✝¹, map_fun' := map_fun …
      ⊢ Eq toEmbedding✝¹ toEmbedding✝
    -/
    ext x
    /-
      case mk.mk.e_toEmbedding.h
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      P : Type u_1
      inst✝¹ : L.Structure P
      Q : Type u_2
      inst✝ : L.Structure Q
      toEmbedding✝¹ : Function.Embedding M N
      map_fun'✝¹ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝ …
      map_rel'✝¹ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder. …
      toEmbedding✝ : Function.Embedding M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝. …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h : Eq ((fun f => f.toFun) { toEmbedding := toEmbedding✝¹, map_fun' := map_fun …
      x : M
      ⊢ Eq (toEmbedding✝¹ x) (toEmbedding✝ x)
    -/
    exact funext_iff.1 h x
    /-
      🎉 no goals
    -/


instance embeddingLike : EmbeddingLike (M ↪[L] N) M N where
  injective' f := f.toEmbedding.injective


instance strongHomClass : StrongHomClass L (M ↪[L] N) M N where
  map_fun := map_fun'
  map_rel := map_rel'


@[simp]
theorem map_fun (φ : M ↪[L] N) {n : ℕ} (f : L.Functions n) (x : Fin n → M) :
    φ (funMap f x) = funMap f (φ ∘ x) :=
  HomClass.map_fun φ f x


@[simp]
theorem map_constants (φ : M ↪[L] N) (c : L.Constants) : φ c = c :=
  HomClass.map_constants φ c


@[simp]
theorem map_rel (φ : M ↪[L] N) {n : ℕ} (r : L.Relations n) (x : Fin n → M) :
    RelMap r (φ ∘ x) ↔ RelMap r x :=
  StrongHomClass.map_rel φ r x


/-- A first-order embedding is also a first-order homomorphism. -/
def toHom : (M ↪[L] N) → M →[L] N :=
  HomClass.toHom


@[simp]
theorem coe_toHom {f : M ↪[L] N} : (f.toHom : M → N) = f :=
  rfl


theorem coe_injective : @Function.Injective (M ↪[L] N) (M → N) (↑)
  | f, g, h => by
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f g : L.Embedding M N
      h : Eq ⇑f ⇑g
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      g : L.Embedding M N
      toEmbedding✝ : Function.Embedding M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝. …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h : Eq ⇑{ toEmbedding := toEmbedding✝, map_fun' := map_fun'✝, map_rel' := map_ …
      ⊢ Eq { toEmbedding := toEmbedding✝, map_fun' := map_fun'✝, map_rel' := map_rel …
    -/
    cases g
    /-
      case mk.mk
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      toEmbedding✝¹ : Function.Embedding M N
      map_fun'✝¹ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝ …
      map_rel'✝¹ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder. …
      toEmbedding✝ : Function.Embedding M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝. …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h : Eq ⇑{ toEmbedding := toEmbedding✝¹, map_fun' := map_fun'✝¹, map_rel' := ma …
      ⊢ Eq { toEmbedding := toEmbedding✝¹, map_fun' := map_fun'✝¹, map_rel' := map_r …
    -/
    congr
    /-
      case mk.mk.e_toEmbedding
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      toEmbedding✝¹ : Function.Embedding M N
      map_fun'✝¹ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝ …
      map_rel'✝¹ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder. …
      toEmbedding✝ : Function.Embedding M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝. …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h : Eq ⇑{ toEmbedding := toEmbedding✝¹, map_fun' := map_fun'✝¹, map_rel' := ma …
      ⊢ Eq toEmbedding✝¹ toEmbedding✝
    -/
    ext x
    /-
      case mk.mk.e_toEmbedding.h
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      toEmbedding✝¹ : Function.Embedding M N
      map_fun'✝¹ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝ …
      map_rel'✝¹ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder. …
      toEmbedding✝ : Function.Embedding M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEmbedding✝. …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h : Eq ⇑{ toEmbedding := toEmbedding✝¹, map_fun' := map_fun'✝¹, map_rel' := ma …
      x : M
      ⊢ Eq (toEmbedding✝¹ x) (toEmbedding✝ x)
    -/
    exact funext_iff.1 h x
    /-
      🎉 no goals
    -/


@[ext]
theorem ext ⦃f g : M ↪[L] N⦄ (h : ∀ x, f x = g x) : f = g :=
  coe_injective (funext h)


theorem toHom_injective : @Function.Injective (M ↪[L] N) (M →[L] N) (·.toHom) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    ⊢ Function.Injective fun x => x.toHom
  -/
  intro f f' h
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f f' : L.Embedding M N
    h : Eq ((fun x => x.toHom) f) ((fun x => x.toHom) f')
    ⊢ Eq f f'
  -/
  ext
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f f' : L.Embedding M N
    h : Eq ((fun x => x.toHom) f) ((fun x => x.toHom) f')
    x✝ : M
    ⊢ Eq (f x✝) (f' x✝)
  -/
  exact congr_fun (congr_arg (↑) h) _
  /-
    🎉 no goals
  -/


@[simp]
theorem toHom_inj {f g : M ↪[L] N} : f.toHom = g.toHom ↔ f = g :=
  ⟨fun h ↦ toHom_injective h, fun h ↦ congr_arg (·.toHom) h⟩


theorem injective (f : M ↪[L] N) : Function.Injective f :=
  f.toEmbedding.injective


/-- In an algebraic language, any injective homomorphism is an embedding. -/
@[simps!]
def ofInjective [L.IsAlgebraic] {f : M →[L] N} (hf : Function.Injective f) : M ↪[L] N :=
  { f with
    inj' := hf
    map_rel' := fun {_} r x => StrongHomClass.map_rel f r x }


@[simp]
theorem coeFn_ofInjective [L.IsAlgebraic] {f : M →[L] N} (hf : Function.Injective f) :
    (ofInjective hf : M → N) = f :=
  rfl


@[simp]
theorem ofInjective_toHom [L.IsAlgebraic] {f : M →[L] N} (hf : Function.Injective f) :
    (ofInjective hf).toHom = f := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    inst✝ : L.IsAlgebraic
    f : L.Hom M N
    hf : Function.Injective ⇑f
    ⊢ Eq (FirstOrder.Language.Embedding.ofInjective hf).toHom f
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- The identity embedding from a structure to itself. -/
@[refl]
def refl : M ↪[L] M where toEmbedding := Function.Embedding.refl M


instance : Inhabited (M ↪[L] M) :=
  ⟨refl L M⟩


@[simp]
theorem refl_apply (x : M) : refl L M x = x :=
  rfl


/-- Composition of first-order embeddings. -/
@[trans]
def comp (hnp : N ↪[L] P) (hmn : M ↪[L] N) : M ↪[L] P where
  toFun := hnp ∘ hmn
  inj' := hnp.injective.comp hmn.injective
  -- Porting note: should be done by autoparam?
                 /-
                   L : FirstOrder.Language
                   L' : FirstOrder.Language
                   M : Type w
                   N : Type w'
                   inst✝³ : L.Structure M
                   inst✝² : L.Structure N
                   P : Type u_1
                   inst✝¹ : L.Structure P
                   Q : Type u_2
                   inst✝ : L.Structure Q
                   hnp : L.Embedding N P
                   hmn : L.Embedding M N
                   ⊢ ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq ({ toFun := Function.com …
                 -/
  map_fun' := by intros; simp only [Function.comp_apply, map_fun]; trivial
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  -- Porting note: should be done by autoparam?
                 /-
                   L : FirstOrder.Language
                   L' : FirstOrder.Language
                   M : Type w
                   N : Type w'
                   inst✝³ : L.Structure M
                   inst✝² : L.Structure N
                   P : Type u_1
                   inst✝¹ : L.Structure P
                   Q : Type u_2
                   inst✝ : L.Structure Q
                   hnp : L.Embedding N P
                   hmn : L.Embedding M N
                   ⊢ ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.Language.St …
                 -/
  map_rel' := by intros; rw [Function.comp_assoc, map_rel, map_rel]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem comp_apply (g : N ↪[L] P) (f : M ↪[L] N) (x : M) : g.comp f x = g (f x) :=
  rfl


/-- Composition of first-order embeddings is associative. -/
theorem comp_assoc (f : M ↪[L] N) (g : N ↪[L] P) (h : P ↪[L] Q) :
    (h.comp g).comp f = h.comp (g.comp f) :=
  rfl


theorem comp_injective (h : N ↪[L] P) :
    Function.Injective (h.comp : (M ↪[L] N) →  (M ↪[L] P)) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    P : Type u_1
    inst✝ : L.Structure P
    h : L.Embedding N P
    ⊢ Function.Injective h.comp
  -/
  intro f g hfg
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    P : Type u_1
    inst✝ : L.Structure P
    h : L.Embedding N P
    f g : L.Embedding M N
    hfg : Eq (h.comp f) (h.comp g)
    ⊢ Eq f g
  -/
  ext x; exact h.injective (DFunLike.congr_fun hfg x)
         /-
           🎉 no goals
         -/


@[simp]
theorem comp_inj (h : N ↪[L] P) (f g : M ↪[L] N) : h.comp f = h.comp g ↔ f = g :=
  ⟨fun eq ↦ h.comp_injective eq, congr_arg h.comp⟩


theorem toHom_comp_injective (h : N ↪[L] P) :
    Function.Injective (h.toHom.comp : (M →[L] N) →  (M →[L] P)) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    P : Type u_1
    inst✝ : L.Structure P
    h : L.Embedding N P
    ⊢ Function.Injective h.toHom.comp
  -/
  intro f g hfg
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    P : Type u_1
    inst✝ : L.Structure P
    h : L.Embedding N P
    f g : L.Hom M N
    hfg : Eq (h.toHom.comp f) (h.toHom.comp g)
    ⊢ Eq f g
  -/
  ext x; exact h.injective (DFunLike.congr_fun hfg x)
         /-
           🎉 no goals
         -/


@[simp]
theorem toHom_comp_inj (h : N ↪[L] P) (f g : M →[L] N) : h.toHom.comp f = h.toHom.comp g ↔ f = g :=
  ⟨fun eq ↦ h.toHom_comp_injective eq, congr_arg h.toHom.comp⟩


@[simp]
theorem comp_toHom (hnp : N ↪[L] P) (hmn : M ↪[L] N) :
    (hnp.comp hmn).toHom = hnp.toHom.comp hmn.toHom :=
  rfl


@[simp]
theorem comp_refl (f : M ↪[L] N) : f.comp (refl L M) = f := DFunLike.coe_injective rfl


@[simp]
theorem refl_comp (f : M ↪[L] N) : (refl L N).comp f = f := DFunLike.coe_injective rfl


@[simp]
theorem refl_toHom : (refl L M).toHom = Hom.id L M :=
  rfl


/-- Any element of an injective `StrongHomClass` can be realized as a first_order embedding. -/
@[simps] def StrongHomClass.toEmbedding {F M N} [L.Structure M] [L.Structure N] [FunLike F M N]
    [EmbeddingLike F M N] [StrongHomClass L F M N] : F → M ↪[L] N := fun φ =>
  ⟨⟨φ, EmbeddingLike.injective φ⟩, StrongHomClass.map_fun φ, StrongHomClass.map_rel φ⟩


instance : EquivLike (M ≃[L] N) M N where
  coe f := f.toFun
  inv f := f.invFun
  left_inv f := f.left_inv
  right_inv f := f.right_inv
  coe_injective' f g h₁ h₂ := by
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      P : Type u_1
      inst✝¹ : L.Structure P
      Q : Type u_2
      inst✝ : L.Structure Q
      f g : L.Equiv M N
      h₁ : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      h₂ : Eq ((fun f => f.invFun) f) ((fun f => f.invFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      P : Type u_1
      inst✝¹ : L.Structure P
      Q : Type u_2
      inst✝ : L.Structure Q
      g : L.Equiv M N
      toEquiv✝ : _root_.Equiv M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEquiv✝.toFu …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h₁ : Eq ((fun f => f.toFun) { toEquiv := toEquiv✝, map_fun' := map_fun'✝, map_ …
      h₂ : Eq ((fun f => f.invFun) { toEquiv := toEquiv✝, map_fun' := map_fun'✝, map …
      ⊢ Eq { toEquiv := toEquiv✝, map_fun' := map_fun'✝, map_rel' := map_rel'✝ } g
    -/
    cases g
    /-
      case mk.mk
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      P : Type u_1
      inst✝¹ : L.Structure P
      Q : Type u_2
      inst✝ : L.Structure Q
      toEquiv✝¹ : _root_.Equiv M N
      map_fun'✝¹ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEquiv✝¹.to …
      map_rel'✝¹ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder. …
      toEquiv✝ : _root_.Equiv M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEquiv✝.toFu …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h₁ : Eq ((fun f => f.toFun) { toEquiv := toEquiv✝¹, map_fun' := map_fun'✝¹, ma …
      h₂ : Eq ((fun f => f.invFun) { toEquiv := toEquiv✝¹, map_fun' := map_fun'✝¹, m …
      ⊢ Eq { toEquiv := toEquiv✝¹, map_fun' := map_fun'✝¹, map_rel' := map_rel'✝¹ }  …
    -/
    simp only [mk.injEq]
    /-
      case mk.mk
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      P : Type u_1
      inst✝¹ : L.Structure P
      Q : Type u_2
      inst✝ : L.Structure Q
      toEquiv✝¹ : _root_.Equiv M N
      map_fun'✝¹ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEquiv✝¹.to …
      map_rel'✝¹ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder. …
      toEquiv✝ : _root_.Equiv M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEquiv✝.toFu …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h₁ : Eq ((fun f => f.toFun) { toEquiv := toEquiv✝¹, map_fun' := map_fun'✝¹, ma …
      h₂ : Eq ((fun f => f.invFun) { toEquiv := toEquiv✝¹, map_fun' := map_fun'✝¹, m …
      ⊢ Eq toEquiv✝¹ toEquiv✝
    -/
    ext x
    /-
      case mk.mk.H
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      P : Type u_1
      inst✝¹ : L.Structure P
      Q : Type u_2
      inst✝ : L.Structure Q
      toEquiv✝¹ : _root_.Equiv M N
      map_fun'✝¹ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEquiv✝¹.to …
      map_rel'✝¹ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder. …
      toEquiv✝ : _root_.Equiv M N
      map_fun'✝ : ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq (toEquiv✝.toFu …
      map_rel'✝ : ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.L …
      h₁ : Eq ((fun f => f.toFun) { toEquiv := toEquiv✝¹, map_fun' := map_fun'✝¹, ma …
      h₂ : Eq ((fun f => f.invFun) { toEquiv := toEquiv✝¹, map_fun' := map_fun'✝¹, m …
      x : M
      ⊢ Eq (toEquiv✝¹ x) (toEquiv✝ x)
    -/
    exact funext_iff.1 h₁ x
    /-
      🎉 no goals
    -/


instance : StrongHomClass L (M ≃[L] N) M N where
  map_fun := map_fun'
  map_rel := map_rel'


/-- The inverse of a first-order equivalence is a first-order equivalence. -/
@[symm]
def symm (f : M ≃[L] N) : N ≃[L] M :=
  { f.toEquiv.symm with
    map_fun' := fun n f' {x} => by
      /-
        L : FirstOrder.Language
        L' : FirstOrder.Language
        M : Type w
        N : Type w'
        inst✝³ : L.Structure M
        inst✝² : L.Structure N
        P : Type u_1
        inst✝¹ : L.Structure P
        Q : Type u_2
        inst✝ : L.Structure Q
        f : L.Equiv M N
        n : Nat
        f' : L.Functions n
        x : Fin n → N
        ⊢ Eq (__src✝.toFun (FirstOrder.Language.Structure.funMap f' x)) (FirstOrder.La …
      -/
      simp only [Equiv.toFun_as_coe]
      /-
        L : FirstOrder.Language
        L' : FirstOrder.Language
        M : Type w
        N : Type w'
        inst✝³ : L.Structure M
        inst✝² : L.Structure N
        P : Type u_1
        inst✝¹ : L.Structure P
        Q : Type u_2
        inst✝ : L.Structure Q
        f : L.Equiv M N
        n : Nat
        f' : L.Functions n
        x : Fin n → N
        ⊢ Eq (f.symm (FirstOrder.Language.Structure.funMap f' x)) (FirstOrder.Language …
      -/
      rw [Equiv.symm_apply_eq]
      /-
        L : FirstOrder.Language
        L' : FirstOrder.Language
        M : Type w
        N : Type w'
        inst✝³ : L.Structure M
        inst✝² : L.Structure N
        P : Type u_1
        inst✝¹ : L.Structure P
        Q : Type u_2
        inst✝ : L.Structure Q
        f : L.Equiv M N
        n : Nat
        f' : L.Functions n
        x : Fin n → N
        ⊢ Eq (FirstOrder.Language.Structure.funMap f' x) (f.toEquiv (FirstOrder.Langua …
      -/
      refine Eq.trans ?_ (f.map_fun' f' (f.toEquiv.symm ∘ x)).symm
      /-
        L : FirstOrder.Language
        L' : FirstOrder.Language
        M : Type w
        N : Type w'
        inst✝³ : L.Structure M
        inst✝² : L.Structure N
        P : Type u_1
        inst✝¹ : L.Structure P
        Q : Type u_2
        inst✝ : L.Structure Q
        f : L.Equiv M N
        n : Nat
        f' : L.Functions n
        x : Fin n → N
        ⊢ Eq (FirstOrder.Language.Structure.funMap f' x) (FirstOrder.Language.Structur …
      -/
      rw [← Function.comp_assoc, Equiv.toFun_as_coe, Equiv.self_comp_symm, Function.id_comp]
      /-
        🎉 no goals
      -/
    map_rel' := fun n r {x} => by
      /-
        L : FirstOrder.Language
        L' : FirstOrder.Language
        M : Type w
        N : Type w'
        inst✝³ : L.Structure M
        inst✝² : L.Structure N
        P : Type u_1
        inst✝¹ : L.Structure P
        Q : Type u_2
        inst✝ : L.Structure Q
        f : L.Equiv M N
        n : Nat
        r : L.Relations n
        x : Fin n → N
        ⊢ Iff (FirstOrder.Language.Structure.RelMap r (Function.comp __src✝.toFun x))  …
      -/
      simp only [Equiv.toFun_as_coe]
      /-
        L : FirstOrder.Language
        L' : FirstOrder.Language
        M : Type w
        N : Type w'
        inst✝³ : L.Structure M
        inst✝² : L.Structure N
        P : Type u_1
        inst✝¹ : L.Structure P
        Q : Type u_2
        inst✝ : L.Structure Q
        f : L.Equiv M N
        n : Nat
        r : L.Relations n
        x : Fin n → N
        ⊢ Iff (FirstOrder.Language.Structure.RelMap r (Function.comp (⇑f.symm) x)) (Fi …
      -/
      refine (f.map_rel' r (f.toEquiv.symm ∘ x)).symm.trans ?_
      /-
        L : FirstOrder.Language
        L' : FirstOrder.Language
        M : Type w
        N : Type w'
        inst✝³ : L.Structure M
        inst✝² : L.Structure N
        P : Type u_1
        inst✝¹ : L.Structure P
        Q : Type u_2
        inst✝ : L.Structure Q
        f : L.Equiv M N
        n : Nat
        r : L.Relations n
        x : Fin n → N
        ⊢ Iff (FirstOrder.Language.Structure.RelMap r (Function.comp f.toFun (Function …
      -/
      rw [← Function.comp_assoc, Equiv.toFun_as_coe, Equiv.self_comp_symm, Function.id_comp] }
      /-
        🎉 no goals
      -/


@[simp]
theorem symm_symm (f : M ≃[L] N) :
    f.symm.symm = f :=
  rfl


@[simp]
theorem apply_symm_apply (f : M ≃[L] N) (a : N) : f (f.symm a) = a :=
  f.toEquiv.apply_symm_apply a


@[simp]
theorem symm_apply_apply (f : M ≃[L] N) (a : M) : f.symm (f a) = a :=
  f.toEquiv.symm_apply_apply a


@[simp]
theorem map_fun (φ : M ≃[L] N) {n : ℕ} (f : L.Functions n) (x : Fin n → M) :
    φ (funMap f x) = funMap f (φ ∘ x) :=
  HomClass.map_fun φ f x


@[simp]
theorem map_constants (φ : M ≃[L] N) (c : L.Constants) : φ c = c :=
  HomClass.map_constants φ c


@[simp]
theorem map_rel (φ : M ≃[L] N) {n : ℕ} (r : L.Relations n) (x : Fin n → M) :
    RelMap r (φ ∘ x) ↔ RelMap r x :=
  StrongHomClass.map_rel φ r x


/-- A first-order equivalence is also a first-order embedding. -/
def toEmbedding : (M ≃[L] N) → M ↪[L] N :=
  StrongHomClass.toEmbedding


/-- A first-order equivalence is also a first-order homomorphism. -/
def toHom : (M ≃[L] N) → M →[L] N :=
  HomClass.toHom


@[simp]
theorem toEmbedding_toHom (f : M ≃[L] N) : f.toEmbedding.toHom = f.toHom :=
  rfl


@[simp]
theorem coe_toHom {f : M ≃[L] N} : (f.toHom : M → N) = (f : M → N) :=
  rfl


@[simp]
theorem coe_toEmbedding (f : M ≃[L] N) : (f.toEmbedding : M → N) = (f : M → N) :=
  rfl


theorem injective_toEmbedding : Function.Injective (toEmbedding : (M ≃[L] N) → M ↪[L] N) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    ⊢ Function.Injective FirstOrder.Language.Equiv.toEmbedding
  -/
  intro _ _ h; apply DFunLike.coe_injective; exact congr_arg (DFunLike.coe ∘ Embedding.toHom) h
                                             /-
                                               🎉 no goals
                                             -/


theorem coe_injective : @Function.Injective (M ≃[L] N) (M → N) (↑) :=
  DFunLike.coe_injective


@[ext]
theorem ext ⦃f g : M ≃[L] N⦄ (h : ∀ x, f x = g x) : f = g :=
  coe_injective (funext h)


theorem bijective (f : M ≃[L] N) : Function.Bijective f :=
  EquivLike.bijective f


theorem injective (f : M ≃[L] N) : Function.Injective f :=
  EquivLike.injective f


theorem surjective (f : M ≃[L] N) : Function.Surjective f :=
  EquivLike.surjective f


/-- The identity equivalence from a structure to itself. -/
@[refl]
def refl : M ≃[L] M where toEquiv := _root_.Equiv.refl M


instance : Inhabited (M ≃[L] M) :=
  ⟨refl L M⟩


@[simp]
                                                  /-
                                                    L : FirstOrder.Language
                                                    M : Type w
                                                    inst✝ : L.Structure M
                                                    x : M
                                                    ⊢ Eq ((FirstOrder.Language.Equiv.refl L M) x) x
                                                  -/
theorem refl_apply (x : M) : refl L M x = x := by simp [refl]; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- Composition of first-order equivalences. -/
@[trans]
def comp (hnp : N ≃[L] P) (hmn : M ≃[L] N) : M ≃[L] P :=
  { hmn.toEquiv.trans hnp.toEquiv with
    toFun := hnp ∘ hmn
    -- Porting note: should be done by autoparam?
                   /-
                     L : FirstOrder.Language
                     L' : FirstOrder.Language
                     M : Type w
                     N : Type w'
                     inst✝³ : L.Structure M
                     inst✝² : L.Structure N
                     P : Type u_1
                     inst✝¹ : L.Structure P
                     Q : Type u_2
                     inst✝ : L.Structure Q
                     hnp : L.Equiv N P
                     hmn : L.Equiv M N
                     ⊢ ∀ {n : Nat} (f : L.Functions n) (x : Fin n → M), Eq ({ toFun := Function.com …
                   -/
    map_fun' := by intros; simp only [Function.comp_apply, map_fun]; trivial
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    -- Porting note: should be done by autoparam?
                   /-
                     L : FirstOrder.Language
                     L' : FirstOrder.Language
                     M : Type w
                     N : Type w'
                     inst✝³ : L.Structure M
                     inst✝² : L.Structure N
                     P : Type u_1
                     inst✝¹ : L.Structure P
                     Q : Type u_2
                     inst✝ : L.Structure Q
                     hnp : L.Equiv N P
                     hmn : L.Equiv M N
                     ⊢ ∀ {n : Nat} (r : L.Relations n) (x : Fin n → M), Iff (FirstOrder.Language.St …
                   -/
    map_rel' := by intros; rw [Function.comp_assoc, map_rel, map_rel] }
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem comp_apply (g : N ≃[L] P) (f : M ≃[L] N) (x : M) : g.comp f x = g (f x) :=
  rfl


@[simp]
theorem comp_refl (g : M ≃[L] N) : g.comp (refl L M) = g :=
  rfl


@[simp]
theorem refl_comp (g : M ≃[L] N) : (refl L N).comp g = g :=
  rfl


@[simp]
theorem refl_toEmbedding : (refl L M).toEmbedding = Embedding.refl L M :=
  rfl


/-- Composition of first-order homomorphisms is associative. -/
theorem comp_assoc (f : M ≃[L] N) (g : N ≃[L] P) (h : P ≃[L] Q) :
    (h.comp g).comp f = h.comp (g.comp f) :=
  rfl


theorem injective_comp (h : N ≃[L] P) :
    Function.Injective (h.comp : (M ≃[L] N) →  (M ≃[L] P)) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    P : Type u_1
    inst✝ : L.Structure P
    h : L.Equiv N P
    ⊢ Function.Injective h.comp
  -/
  intro f g hfg
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    P : Type u_1
    inst✝ : L.Structure P
    h : L.Equiv N P
    f g : L.Equiv M N
    hfg : Eq (h.comp f) (h.comp g)
    ⊢ Eq f g
  -/
  ext x; exact h.injective (congr_fun (congr_arg DFunLike.coe hfg) x)
         /-
           🎉 no goals
         -/


@[simp]
theorem comp_toHom (hnp : N ≃[L] P) (hmn : M ≃[L] N) :
    (hnp.comp hmn).toHom = hnp.toHom.comp hmn.toHom :=
  rfl


@[simp]
theorem comp_toEmbedding (hnp : N ≃[L] P) (hmn : M ≃[L] N) :
    (hnp.comp hmn).toEmbedding = hnp.toEmbedding.comp hmn.toEmbedding :=
  rfl


@[simp]
theorem self_comp_symm (f : M ≃[L] N) : f.comp f.symm = refl L N := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Equiv M N
    ⊢ Eq (f.comp f.symm) (FirstOrder.Language.Equiv.refl L N)
  -/
  ext; rw [comp_apply, apply_symm_apply, refl_apply]
       /-
         🎉 no goals
       -/


@[simp]
theorem symm_comp_self (f : M ≃[L] N) : f.symm.comp f = refl L M := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Equiv M N
    ⊢ Eq (f.symm.comp f) (FirstOrder.Language.Equiv.refl L M)
  -/
  ext; rw [comp_apply, symm_apply_apply, refl_apply]
       /-
         🎉 no goals
       -/


@[simp]
theorem symm_comp_self_toEmbedding (f : M ≃[L] N) :
    f.symm.toEmbedding.comp f.toEmbedding = Embedding.refl L M := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Equiv M N
    ⊢ Eq (f.symm.toEmbedding.comp f.toEmbedding) (FirstOrder.Language.Embedding.re …
  -/
  rw [← comp_toEmbedding, symm_comp_self, refl_toEmbedding]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_comp_symm_toEmbedding (f : M ≃[L] N) :
    f.toEmbedding.comp f.symm.toEmbedding = Embedding.refl L N := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Equiv M N
    ⊢ Eq (f.toEmbedding.comp f.symm.toEmbedding) (FirstOrder.Language.Embedding.re …
  -/
  rw [← comp_toEmbedding, self_comp_symm, refl_toEmbedding]
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_comp_self_toHom (f : M ≃[L] N) :
    f.symm.toHom.comp f.toHom = Hom.id L M := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Equiv M N
    ⊢ Eq (f.symm.toHom.comp f.toHom) (FirstOrder.Language.Hom.id L M)
  -/
  rw [← comp_toHom, symm_comp_self, refl_toHom]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_comp_symm_toHom (f : M ≃[L] N) :
    f.toHom.comp f.symm.toHom = Hom.id L N := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Equiv M N
    ⊢ Eq (f.toHom.comp f.symm.toHom) (FirstOrder.Language.Hom.id L N)
  -/
  rw [← comp_toHom, self_comp_symm, refl_toHom]
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_symm (f : M ≃[L] N) (g : N ≃[L] P) : (g.comp f).symm = f.symm.comp g.symm :=
  rfl


theorem comp_right_injective (h : M ≃[L] N) :
    Function.Injective (fun f ↦ f.comp h : (N ≃[L] P) → (M ≃[L] P)) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    P : Type u_1
    inst✝ : L.Structure P
    h : L.Equiv M N
    ⊢ Function.Injective fun f => f.comp h
  -/
  intro f g hfg
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    P : Type u_1
    inst✝ : L.Structure P
    h : L.Equiv M N
    f g : L.Equiv N P
    hfg : Eq ((fun f => f.comp h) f) ((fun f => f.comp h) g)
    ⊢ Eq f g
  -/
  convert (congr_arg (fun r : (M ≃[L] P) ↦ r.comp h.symm) hfg) <;>
    /-
      case h.e'_2
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝² : L.Structure M
      inst✝¹ : L.Structure N
      P : Type u_1
      inst✝ : L.Structure P
      h : L.Equiv M N
      f g : L.Equiv N P
      hfg : Eq ((fun f => f.comp h) f) ((fun f => f.comp h) g)
      ⊢ Eq f (((fun f => f.comp h) f).comp h.symm)
    -/
    /-
      🎉 no goals
    -/
    rw [comp_assoc, self_comp_symm, comp_refl]
    /-
      🎉 no goals
    -/


@[simp]
theorem comp_right_inj (h : M ≃[L] N) (f g : N ≃[L] P) : f.comp h = g.comp h ↔ f = g :=
  ⟨fun eq ↦ h.comp_right_injective eq, congr_arg (fun (r : N ≃[L] P) ↦ r.comp h)⟩


/-- Any element of a bijective `StrongHomClass` can be realized as a first_order isomorphism. -/
@[simps] def StrongHomClass.toEquiv {F M N} [L.Structure M] [L.Structure N] [EquivLike F M N]
    [StrongHomClass L F M N] : F → M ≃[L] N := fun φ =>
  ⟨⟨φ, EquivLike.inv φ, EquivLike.left_inv φ, EquivLike.right_inv φ⟩, StrongHomClass.map_fun φ,
    StrongHomClass.map_rel φ⟩


instance sumStructure : (L₁.sum L₂).Structure S where
  funMap := Sum.elim funMap funMap
  RelMap := Sum.elim RelMap RelMap


@[simp]
theorem funMap_sum_inl {n : ℕ} (f : L₁.Functions n) :
    @funMap (L₁.sum L₂) S _ n (Sum.inl f) = funMap f :=
  rfl


@[simp]
theorem funMap_sum_inr {n : ℕ} (f : L₂.Functions n) :
    @funMap (L₁.sum L₂) S _ n (Sum.inr f) = funMap f :=
  rfl


@[simp]
theorem relMap_sum_inl {n : ℕ} (R : L₁.Relations n) :
    @RelMap (L₁.sum L₂) S _ n (Sum.inl R) = RelMap R :=
  rfl


@[simp]
theorem relMap_sum_inr {n : ℕ} (R : L₂.Relations n) :
    @RelMap (L₁.sum L₂) S _ n (Sum.inr R) = RelMap R :=
  rfl


/-- Any type can be made uniquely into a structure over the empty language. -/
def emptyStructure : Language.empty.Structure M where


instance : Unique (Language.empty.Structure M) :=
  ⟨⟨Language.emptyStructure⟩, fun a => by
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      P : Type u_1
      inst✝¹ : L.Structure P
      Q : Type u_2
      inst✝ : L.Structure Q
      a : FirstOrder.Language.empty.Structure M
      ⊢ Eq a Inhabited.default
    -/
                /-
                  🎉 no goals
                -/
    ext _ f <;> exact Empty.elim f⟩
                /-
                  🎉 no goals
                -/


instance (priority := 100) strongHomClassEmpty {F} [FunLike F M N] :
    StrongHomClass Language.empty F M N :=
  ⟨fun _ _ f => Empty.elim f, fun _ _ r => Empty.elim r⟩


@[simp]
theorem empty.nonempty_embedding_iff :
    Nonempty (M ↪[Language.empty] N) ↔ Cardinal.lift.{w'} #M ≤ Cardinal.lift.{w} #N :=
  _root_.trans ⟨Nonempty.map fun f => f.toEmbedding, Nonempty.map StrongHomClass.toEmbedding⟩
    Cardinal.lift_mk_le'.symm


@[simp]
theorem empty.nonempty_equiv_iff :
    Nonempty (M ≃[Language.empty] N) ↔ Cardinal.lift.{w'} #M = Cardinal.lift.{w} #N :=
  _root_.trans ⟨Nonempty.map fun f => f.toEquiv, Nonempty.map fun f => { toEquiv := f }⟩
    Cardinal.lift_mk_eq'.symm


/-- Makes a `Language.empty.Hom` out of any function.
This is only needed because there is no instance of `FunLike (M → N) M N`, and thus no instance of
`Language.empty.HomClass M N`. -/
@[simps]
def _root_.Function.emptyHom (f : M → N) : M →[Language.empty] N where toFun := f


/-- A structure induced by a bijection. -/
@[simps!]
def inducedStructure (e : M ≃ N) : L.Structure N :=
  ⟨fun f x => e (funMap f (e.symm ∘ x)), fun r x => RelMap r (e.symm ∘ x)⟩


/-- A bijection as a first-order isomorphism with the induced structure on the codomain. -/
--@[simps!] Porting note: commented out and lemmas added manually
def inducedStructureEquiv (e : M ≃ N) : @Language.Equiv L M N _ (inducedStructure e) := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    N : Type u_2
    inst✝ : L.Structure M
    e : Equiv M N
    ⊢ L.Equiv M N
  -/
  letI : L.Structure N := inducedStructure e
  exact
  { e with
    map_fun' := @fun n f x => by simp [← Function.comp_assoc e.symm e x]
    map_rel' := @fun n r x => by simp [← Function.comp_assoc e.symm e x] }


@[simp]
theorem toEquiv_inducedStructureEquiv (e : M ≃ N) :
    @Language.Equiv.toEquiv L M N _ (inducedStructure e) (inducedStructureEquiv e) = e :=
  rfl


@[simp]
theorem toFun_inducedStructureEquiv (e : M ≃ N) :
    DFunLike.coe (@inducedStructureEquiv L M N _ e) = e :=
  rfl


@[simp]
theorem toFun_inducedStructureEquiv_Symm (e : M ≃ N) :
    (by
    /-
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝ : L.Structure M
      e : Equiv M N
      ⊢ N → M
    -/
    letI : L.Structure N := inducedStructure e
    /-
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝ : L.Structure M
      e : Equiv M N
      this : L.Structure N := e.inducedStructure
      ⊢ N → M
    -/
    exact DFunLike.coe (@inducedStructureEquiv L M N _ e).symm) = (e.symm : N → M) :=
    /-
      🎉 no goals
    -/
  rfl


