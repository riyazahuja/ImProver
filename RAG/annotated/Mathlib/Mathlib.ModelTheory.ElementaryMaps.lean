/-- An elementary embedding of first-order structures is an embedding that commutes with the
  realizations of formulas. -/
structure ElementaryEmbedding where
  toFun : M → N
  -- Porting note:
  -- The autoparam here used to be `obviously`. We would like to replace it with `aesop`
  -- but that isn't currently sufficient.
  -- See https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/Aesop.20and.20cases
  -- If that can be improved, we should change this to `by aesop` and remove the proofs below.
  map_formula' :
    ∀ ⦃n⦄ (φ : L.Formula (Fin n)) (x : Fin n → M), φ.Realize (toFun ∘ x) ↔ φ.Realize x := by
    intros; trivial


@[inherit_doc FirstOrder.Language.ElementaryEmbedding]
scoped[FirstOrder] notation:25 A " ↪ₑ[" L "] " B => FirstOrder.Language.ElementaryEmbedding L A B


instance instFunLike : FunLike (M ↪ₑ[L] N) M N where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      P : Type u_3
      Q : Type u_4
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      inst✝¹ : L.Structure P
      inst✝ : L.Structure Q
      f g : L.ElementaryEmbedding M N
      h : Eq ((fun f => ↑f) f) ((fun f => ↑f) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      P : Type u_3
      Q : Type u_4
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      inst✝¹ : L.Structure P
      inst✝ : L.Structure Q
      g : L.ElementaryEmbedding M N
      toFun✝ : M → N
      map_formula'✝ : ∀ ⦃n : Nat⦄ (φ : L.Formula (Fin n)) (x : Fin n → M), Iff (φ.Re …
      h : Eq ((fun f => ↑f) { toFun := toFun✝, map_formula' := map_formula'✝ }) ((fu …
      ⊢ Eq { toFun := toFun✝, map_formula' := map_formula'✝ } g
    -/
    cases g
    /-
      case mk.mk
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      P : Type u_3
      Q : Type u_4
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      inst✝¹ : L.Structure P
      inst✝ : L.Structure Q
      toFun✝¹ : M → N
      map_formula'✝¹ : ∀ ⦃n : Nat⦄ (φ : L.Formula (Fin n)) (x : Fin n → M), Iff (φ.R …
      toFun✝ : M → N
      map_formula'✝ : ∀ ⦃n : Nat⦄ (φ : L.Formula (Fin n)) (x : Fin n → M), Iff (φ.Re …
      h : Eq ((fun f => ↑f) { toFun := toFun✝¹, map_formula' := map_formula'✝¹ }) (( …
      ⊢ Eq { toFun := toFun✝¹, map_formula' := map_formula'✝¹ } { toFun := toFun✝, m …
    -/
    simp only [ElementaryEmbedding.mk.injEq]
    /-
      case mk.mk
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      P : Type u_3
      Q : Type u_4
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      inst✝¹ : L.Structure P
      inst✝ : L.Structure Q
      toFun✝¹ : M → N
      map_formula'✝¹ : ∀ ⦃n : Nat⦄ (φ : L.Formula (Fin n)) (x : Fin n → M), Iff (φ.R …
      toFun✝ : M → N
      map_formula'✝ : ∀ ⦃n : Nat⦄ (φ : L.Formula (Fin n)) (x : Fin n → M), Iff (φ.Re …
      h : Eq ((fun f => ↑f) { toFun := toFun✝¹, map_formula' := map_formula'✝¹ }) (( …
      ⊢ Eq toFun✝¹ toFun✝
    -/
    ext x
    /-
      case mk.mk.h
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      P : Type u_3
      Q : Type u_4
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      inst✝¹ : L.Structure P
      inst✝ : L.Structure Q
      toFun✝¹ : M → N
      map_formula'✝¹ : ∀ ⦃n : Nat⦄ (φ : L.Formula (Fin n)) (x : Fin n → M), Iff (φ.R …
      toFun✝ : M → N
      map_formula'✝ : ∀ ⦃n : Nat⦄ (φ : L.Formula (Fin n)) (x : Fin n → M), Iff (φ.Re …
      h : Eq ((fun f => ↑f) { toFun := toFun✝¹, map_formula' := map_formula'✝¹ }) (( …
      x : M
      ⊢ Eq (toFun✝¹ x) (toFun✝ x)
    -/
    exact funext_iff.1 h x
    /-
      🎉 no goals
    -/


@[simp]
theorem map_boundedFormula (f : M ↪ₑ[L] N) {α : Type*} {n : ℕ} (φ : L.BoundedFormula α n)
    (v : α → M) (xs : Fin n → M) : φ.Realize (f ∘ v) (f ∘ xs) ↔ φ.Realize v xs := by
  classical
    rw [← BoundedFormula.realize_restrictFreeVar' Set.Subset.rfl, Set.inclusion_eq_id, iff_eq_eq]
    have h :=
      f.map_formula' ((φ.restrictFreeVar id).toFormula.relabel (Fintype.equivFin _))
        (Sum.elim (v ∘ (↑)) xs ∘ (Fintype.equivFin _).symm)
    simp only [Formula.realize_relabel, BoundedFormula.realize_toFormula, iff_eq_eq] at h
    rw [← Function.comp_assoc _ _ (Fintype.equivFin _).symm,
      Function.comp_assoc _ (Fintype.equivFin _).symm (Fintype.equivFin _),
      _root_.Equiv.symm_comp_self, Function.comp_id, Function.comp_assoc, Sum.elim_comp_inl,
      Function.comp_assoc _ _ Sum.inr, Sum.elim_comp_inr, ← Function.comp_assoc] at h
    refine h.trans ?_
    erw [Function.comp_assoc _ _ (Fintype.equivFin _), _root_.Equiv.symm_comp_self,
      Function.comp_id, Sum.elim_comp_inl, Sum.elim_comp_inr (v ∘ Subtype.val) xs,
      ← Set.inclusion_eq_id (s := (BoundedFormula.freeVarFinset φ : Set α)) Set.Subset.rfl,
      BoundedFormula.realize_restrictFreeVar' Set.Subset.rfl]


@[simp]
theorem map_formula (f : M ↪ₑ[L] N) {α : Type*} (φ : L.Formula α) (x : α → M) :
    φ.Realize (f ∘ x) ↔ φ.Realize x := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    N : Type u_2
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.ElementaryEmbedding M N
    α : Type u_5
    φ : L.Formula α
    x : α → M
    ⊢ Iff (φ.Realize (Function.comp (⇑f) x)) (φ.Realize x)
  -/
  rw [Formula.Realize, Formula.Realize, ← f.map_boundedFormula, Unique.eq_default (f ∘ default)]
  /-
    🎉 no goals
  -/


theorem map_sentence (f : M ↪ₑ[L] N) (φ : L.Sentence) : M ⊨ φ ↔ N ⊨ φ := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    N : Type u_2
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.ElementaryEmbedding M N
    φ : L.Sentence
    ⊢ Iff (FirstOrder.Language.Sentence.Realize M φ) (FirstOrder.Language.Sentence …
  -/
  rw [Sentence.Realize, Sentence.Realize, ← f.map_formula, Unique.eq_default (f ∘ default)]
  /-
    🎉 no goals
  -/


theorem theory_model_iff (f : M ↪ₑ[L] N) (T : L.Theory) : M ⊨ T ↔ N ⊨ T := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    N : Type u_2
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.ElementaryEmbedding M N
    T : L.Theory
    ⊢ Iff (FirstOrder.Language.Theory.Model M T) (FirstOrder.Language.Theory.Model …
  -/
  simp only [Theory.model_iff, f.map_sentence]
  /-
    🎉 no goals
  -/


theorem elementarilyEquivalent (f : M ↪ₑ[L] N) : M ≅[L] N :=
  elementarilyEquivalent_iff.2 f.map_sentence


@[simp]
theorem injective (φ : M ↪ₑ[L] N) : Function.Injective φ := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    N : Type u_2
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    φ : L.ElementaryEmbedding M N
    ⊢ Function.Injective ⇑φ
  -/
  intro x y
  have h :=
    φ.map_formula ((var 0).equal (var 1) : L.Formula (Fin 2)) fun i => if i = 0 then x else y
  /-
    L : FirstOrder.Language
    M : Type u_1
    N : Type u_2
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    φ : L.ElementaryEmbedding M N
    x y : M
    h : Iff (((FirstOrder.Language.Term.var 0).equal (FirstOrder.Language.Term.var …
    ⊢ Eq (φ x) (φ y) → Eq x y
  -/
  rw [Formula.realize_equal, Formula.realize_equal] at h
  simp only [Nat.one_ne_zero, Term.realize, Fin.one_eq_zero_iff, if_true, eq_self_iff_true,
    Function.comp_apply, if_false] at h
  /-
    L : FirstOrder.Language
    M : Type u_1
    N : Type u_2
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    φ : L.ElementaryEmbedding M N
    x y : M
    h : Iff (Eq (φ x) (φ (ite (Eq 2 1) x y))) (Eq x (ite (Eq 2 1) x y))
    ⊢ Eq (φ x) (φ y) → Eq x y
  -/
  exact h.1
  /-
    🎉 no goals
  -/


instance embeddingLike : EmbeddingLike (M ↪ₑ[L] N) M N :=
  { show FunLike (M ↪ₑ[L] N) M N from inferInstance with injective' := injective }


@[simp]
theorem map_fun (φ : M ↪ₑ[L] N) {n : ℕ} (f : L.Functions n) (x : Fin n → M) :
    φ (funMap f x) = funMap f (φ ∘ x) := by
  /-
    L : FirstOrder.Language
    M : Type u_1
    N : Type u_2
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    φ : L.ElementaryEmbedding M N
    n : Nat
    f : L.Functions n
    x : Fin n → M
    ⊢ Eq (φ (FirstOrder.Language.Structure.funMap f x)) (FirstOrder.Language.Struc …
  -/
  have h := φ.map_formula (Formula.graph f) (Fin.cons (funMap f x) x)
  /-
    L : FirstOrder.Language
    M : Type u_1
    N : Type u_2
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    φ : L.ElementaryEmbedding M N
    n : Nat
    f : L.Functions n
    x : Fin n → M
    h : Iff ((FirstOrder.Language.Formula.graph f).Realize (Function.comp (⇑φ) (Fi …
    ⊢ Eq (φ (FirstOrder.Language.Structure.funMap f x)) (FirstOrder.Language.Struc …
  -/
  rw [Formula.realize_graph, Fin.comp_cons, Formula.realize_graph] at h
  /-
    L : FirstOrder.Language
    M : Type u_1
    N : Type u_2
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    φ : L.ElementaryEmbedding M N
    n : Nat
    f : L.Functions n
    x : Fin n → M
    h : Iff (Eq (FirstOrder.Language.Structure.funMap f (Function.comp (⇑φ) x)) (φ …
    ⊢ Eq (φ (FirstOrder.Language.Structure.funMap f x)) (FirstOrder.Language.Struc …
  -/
  rw [eq_comm, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_rel (φ : M ↪ₑ[L] N) {n : ℕ} (r : L.Relations n) (x : Fin n → M) :
    RelMap r (φ ∘ x) ↔ RelMap r x :=
  haveI h := φ.map_formula (r.formula var) x
  h


instance strongHomClass : StrongHomClass L (M ↪ₑ[L] N) M N where
  map_fun := map_fun
  map_rel := map_rel


@[simp]
theorem map_constants (φ : M ↪ₑ[L] N) (c : L.Constants) : φ c = c :=
  HomClass.map_constants φ c


/-- An elementary embedding is also a first-order embedding. -/
def toEmbedding (f : M ↪ₑ[L] N) : M ↪[L] N where
  toFun := f
  inj' := f.injective
                         /-
                           L : FirstOrder.Language
                           M : Type u_1
                           N : Type u_2
                           P : Type u_3
                           Q : Type u_4
                           inst✝³ : L.Structure M
                           inst✝² : L.Structure N
                           inst✝¹ : L.Structure P
                           inst✝ : L.Structure Q
                           f✝ : L.ElementaryEmbedding M N
                           x✝ : Nat
                           f : L.Functions x✝
                           x : Fin x✝ → M
                           ⊢ Eq ({ toFun := ⇑f✝, inj' := ⋯ }.toFun (FirstOrder.Language.Structure.funMap  …
                         -/
  map_fun' {_} f x := by aesop
                         /-
                           🎉 no goals
                         -/
                         /-
                           L : FirstOrder.Language
                           M : Type u_1
                           N : Type u_2
                           P : Type u_3
                           Q : Type u_4
                           inst✝³ : L.Structure M
                           inst✝² : L.Structure N
                           inst✝¹ : L.Structure P
                           inst✝ : L.Structure Q
                           f : L.ElementaryEmbedding M N
                           x✝ : Nat
                           R : L.Relations x✝
                           x : Fin x✝ → M
                           ⊢ Iff (FirstOrder.Language.Structure.RelMap R (Function.comp { toFun := ⇑f, in …
                         -/
  map_rel' {_} R x := by aesop
                         /-
                           🎉 no goals
                         -/


/-- An elementary embedding is also a first-order homomorphism. -/
def toHom (f : M ↪ₑ[L] N) : M →[L] N where
  toFun := f
                         /-
                           L : FirstOrder.Language
                           M : Type u_1
                           N : Type u_2
                           P : Type u_3
                           Q : Type u_4
                           inst✝³ : L.Structure M
                           inst✝² : L.Structure N
                           inst✝¹ : L.Structure P
                           inst✝ : L.Structure Q
                           f✝ : L.ElementaryEmbedding M N
                           x✝ : Nat
                           f : L.Functions x✝
                           x : Fin x✝ → M
                           ⊢ Eq (f✝ (FirstOrder.Language.Structure.funMap f x)) (FirstOrder.Language.Stru …
                         -/
  map_fun' {_} f x := by aesop
                         /-
                           🎉 no goals
                         -/
                         /-
                           L : FirstOrder.Language
                           M : Type u_1
                           N : Type u_2
                           P : Type u_3
                           Q : Type u_4
                           inst✝³ : L.Structure M
                           inst✝² : L.Structure N
                           inst✝¹ : L.Structure P
                           inst✝ : L.Structure Q
                           f : L.ElementaryEmbedding M N
                           x✝ : Nat
                           R : L.Relations x✝
                           x : Fin x✝ → M
                           ⊢ FirstOrder.Language.Structure.RelMap R x → FirstOrder.Language.Structure.Rel …
                         -/
  map_rel' {_} R x := by aesop
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem toEmbedding_toHom (f : M ↪ₑ[L] N) : f.toEmbedding.toHom = f.toHom :=
  rfl


@[simp]
theorem coe_toHom {f : M ↪ₑ[L] N} : (f.toHom : M → N) = (f : M → N) :=
  rfl


@[simp]
theorem coe_toEmbedding (f : M ↪ₑ[L] N) : (f.toEmbedding : M → N) = (f : M → N) :=
  rfl


theorem coe_injective : @Function.Injective (M ↪ₑ[L] N) (M → N) (↑) :=
  DFunLike.coe_injective


@[ext]
theorem ext ⦃f g : M ↪ₑ[L] N⦄ (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


/-- The identity elementary embedding from a structure to itself -/
@[refl]
def refl : M ↪ₑ[L] M where toFun := id


instance : Inhabited (M ↪ₑ[L] M) :=
  ⟨refl L M⟩


@[simp]
theorem refl_apply (x : M) : refl L M x = x :=
  rfl


/-- Composition of elementary embeddings -/
@[trans]
def comp (hnp : N ↪ₑ[L] P) (hmn : M ↪ₑ[L] N) : M ↪ₑ[L] P where
  toFun := hnp ∘ hmn
  map_formula' n φ x := by
    /-
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      P : Type u_3
      Q : Type u_4
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      inst✝¹ : L.Structure P
      inst✝ : L.Structure Q
      hnp : L.ElementaryEmbedding N P
      hmn : L.ElementaryEmbedding M N
      n : Nat
      φ : L.Formula (Fin n)
      x : Fin n → M
      ⊢ Iff (φ.Realize (Function.comp (Function.comp ⇑hnp ⇑hmn) x)) (φ.Realize x)
    -/
    cases' hnp with _ hhnp
    /-
      case mk
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      P : Type u_3
      Q : Type u_4
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      inst✝¹ : L.Structure P
      inst✝ : L.Structure Q
      hmn : L.ElementaryEmbedding M N
      n : Nat
      φ : L.Formula (Fin n)
      x : Fin n → M
      toFun✝ : N → P
      hhnp : ∀ ⦃n : Nat⦄ (φ : L.Formula (Fin n)) (x : Fin n → N), Iff (φ.Realize (Fu …
      ⊢ Iff (φ.Realize (Function.comp (Function.comp ⇑{ toFun := toFun✝, map_formula …
    -/
    cases' hmn with _ hhmn
    /-
      case mk.mk
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      P : Type u_3
      Q : Type u_4
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      inst✝¹ : L.Structure P
      inst✝ : L.Structure Q
      n : Nat
      φ : L.Formula (Fin n)
      x : Fin n → M
      toFun✝¹ : N → P
      hhnp : ∀ ⦃n : Nat⦄ (φ : L.Formula (Fin n)) (x : Fin n → N), Iff (φ.Realize (Fu …
      toFun✝ : M → N
      hhmn : ∀ ⦃n : Nat⦄ (φ : L.Formula (Fin n)) (x : Fin n → M), Iff (φ.Realize (Fu …
      ⊢ Iff (φ.Realize (Function.comp (Function.comp ⇑{ toFun := toFun✝¹, map_formul …
    -/
    erw [hhnp, hhmn]
    /-
      🎉 no goals
    -/


@[simp]
theorem comp_apply (g : N ↪ₑ[L] P) (f : M ↪ₑ[L] N) (x : M) : g.comp f x = g (f x) :=
  rfl


/-- Composition of elementary embeddings is associative. -/
theorem comp_assoc (f : M ↪ₑ[L] N) (g : N ↪ₑ[L] P) (h : P ↪ₑ[L] Q) :
    (h.comp g).comp f = h.comp (g.comp f) :=
  rfl


/-- The elementary diagram of an `L`-structure is the set of all sentences with parameters it
  satisfies. -/
abbrev elementaryDiagram : L[[M]].Theory :=
  L[[M]].completeTheory M


/-- The canonical elementary embedding of an `L`-structure into any model of its elementary diagram
-/
@[simps]
def ElementaryEmbedding.ofModelsElementaryDiagram (N : Type*) [L.Structure N] [L[[M]].Structure N]
    [(lhomWithConstants L M).IsExpansionOn N] [N ⊨ L.elementaryDiagram M] : M ↪ₑ[L] N :=
  ⟨((↑) : L[[M]].Constants → N) ∘ Sum.inr, fun n φ x => by
    refine
      _root_.trans ?_
        ((realize_iff_of_model_completeTheory M N
              (((L.lhomWithConstants M).onBoundedFormula φ).subst
                  (Constants.term ∘ Sum.inr ∘ x)).alls).trans
          ?_)
    · simp_rw [Sentence.Realize, BoundedFormula.realize_alls, BoundedFormula.realize_subst,
        LHom.realize_onBoundedFormula, Formula.Realize, Unique.forall_iff, Function.comp_def,
        Term.realize_constants]
    · simp_rw [Sentence.Realize, BoundedFormula.realize_alls, BoundedFormula.realize_subst,
        LHom.realize_onBoundedFormula, Formula.Realize, Unique.forall_iff]
      /-
        case refine_2
        L : FirstOrder.Language
        M : Type u_1
        N✝ : Type u_2
        P : Type u_3
        Q : Type u_4
        inst✝⁷ : L.Structure M
        inst✝⁶ : L.Structure N✝
        inst✝⁵ : L.Structure P
        inst✝⁴ : L.Structure Q
        N : Type u_5
        inst✝³ : L.Structure N
        inst✝² : (L.withConstants M).Structure N
        inst✝¹ : (L.lhomWithConstants M).IsExpansionOn N
        inst✝ : FirstOrder.Language.Theory.Model N (L.elementaryDiagram M)
        n : Nat
        φ : L.Formula (Fin n)
        x : Fin n → M
        ⊢ Iff (FirstOrder.Language.BoundedFormula.Realize φ (fun a => FirstOrder.Langu …
      -/
      rfl⟩
      /-
        🎉 no goals
      -/


/-- The **Tarski-Vaught test** for elementarity of an embedding. -/
theorem isElementary_of_exists (f : M ↪[L] N)
    (htv :
      ∀ (n : ℕ) (φ : L.BoundedFormula Empty (n + 1)) (x : Fin n → M) (a : N),
        φ.Realize default (Fin.snoc (f ∘ x) a : _ → N) →
          ∃ b : M, φ.Realize default (Fin.snoc (f ∘ x) (f b) : _ → N)) :
    ∀ {n} (φ : L.Formula (Fin n)) (x : Fin n → M), φ.Realize (f ∘ x) ↔ φ.Realize x := by
  suffices h : ∀ (n : ℕ) (φ : L.BoundedFormula Empty n) (xs : Fin n → M),
      φ.Realize (f ∘ default) (f ∘ xs) ↔ φ.Realize default xs by
    intro n φ x
    exact φ.realize_relabel_sum_inr.symm.trans (_root_.trans (h n _ _) φ.realize_relabel_sum_inr)
  /-
    L : FirstOrder.Language
    M : Type u_1
    N : Type u_2
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Embedding M N
    htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
    ⊢ ∀ (n : Nat) (φ : L.BoundedFormula Empty n) (xs : Fin n → M), Iff (φ.Realize  …
  -/
  refine fun n φ => φ.recOn ?_ ?_ ?_ ?_ ?_
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f : L.Embedding M N
      htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
      n : Nat
      φ : L.BoundedFormula Empty n
      ⊢ ∀ {n : Nat} (xs : Fin n → M), Iff (FirstOrder.Language.BoundedFormula.falsum …
    -/
  · exact fun {_} _ => Iff.rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f : L.Embedding M N
      htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
      n : Nat
      φ : L.BoundedFormula Empty n
      ⊢ ∀ {n : Nat} (t₁ t₂ : L.Term (Sum Empty (Fin n))) (xs : Fin n → M), Iff ((Fir …
    -/
  · intros
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f : L.Embedding M N
      htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
      n : Nat
      φ : L.BoundedFormula Empty n
      n✝ : Nat
      t₁✝ t₂✝ : L.Term (Sum Empty (Fin n✝))
      xs✝ : Fin n✝ → M
      ⊢ Iff ((FirstOrder.Language.BoundedFormula.equal t₁✝ t₂✝).Realize (Function.co …
    -/
    simp [BoundedFormula.Realize, ← Sum.comp_elim, HomClass.realize_term]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f : L.Embedding M N
      htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
      n : Nat
      φ : L.BoundedFormula Empty n
      ⊢ ∀ {n l : Nat} (R : L.Relations l) (ts : Fin l → L.Term (Sum Empty (Fin n)))  …
    -/
  · intros
    /-
      case refine_3
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f : L.Embedding M N
      htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
      n : Nat
      φ : L.BoundedFormula Empty n
      n✝ l✝ : Nat
      R✝ : L.Relations l✝
      ts✝ : Fin l✝ → L.Term (Sum Empty (Fin n✝))
      xs✝ : Fin n✝ → M
      ⊢ Iff ((FirstOrder.Language.BoundedFormula.rel R✝ ts✝).Realize (Function.comp  …
    -/
    simp only [BoundedFormula.Realize, ← Sum.comp_elim, HomClass.realize_term]
    /-
      case refine_3
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f : L.Embedding M N
      htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
      n : Nat
      φ : L.BoundedFormula Empty n
      n✝ l✝ : Nat
      R✝ : L.Relations l✝
      ts✝ : Fin l✝ → L.Term (Sum Empty (Fin n✝))
      xs✝ : Fin n✝ → M
      ⊢ Iff (FirstOrder.Language.Structure.RelMap R✝ fun i => f (FirstOrder.Language …
    -/
    erw [map_rel f]
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f : L.Embedding M N
      htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
      n : Nat
      φ : L.BoundedFormula Empty n
      ⊢ ∀ {n : Nat} (f₁ f₂ : L.BoundedFormula Empty n), (∀ (xs : Fin n → M), Iff (f₁ …
    -/
  · intro _ _ _ ih1 ih2 _
    /-
      case refine_4
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f : L.Embedding M N
      htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
      n : Nat
      φ : L.BoundedFormula Empty n
      n✝ : Nat
      f₁✝ f₂✝ : L.BoundedFormula Empty n✝
      ih1 : ∀ (xs : Fin n✝ → M), Iff (f₁✝.Realize (Function.comp (⇑f) Inhabited.defa …
      ih2 : ∀ (xs : Fin n✝ → M), Iff (f₂✝.Realize (Function.comp (⇑f) Inhabited.defa …
      xs✝ : Fin n✝ → M
      ⊢ Iff ((f₁✝.imp f₂✝).Realize (Function.comp (⇑f) Inhabited.default) (Function. …
    -/
    simp [ih1, ih2]
    /-
      🎉 no goals
    -/
    /-
      case refine_5
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f : L.Embedding M N
      htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
      n : Nat
      φ : L.BoundedFormula Empty n
      ⊢ ∀ {n : Nat} (f_1 : L.BoundedFormula Empty (HAdd.hAdd n 1)), (∀ (xs : Fin (HA …
    -/
  · intro n φ ih xs
    /-
      case refine_5
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f : L.Embedding M N
      htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
      n✝ : Nat
      φ✝ : L.BoundedFormula Empty n✝
      n : Nat
      φ : L.BoundedFormula Empty (HAdd.hAdd n 1)
      ih : ∀ (xs : Fin (HAdd.hAdd n 1) → M), Iff (φ.Realize (Function.comp (⇑f) Inha …
      xs : Fin n → M
      ⊢ Iff (φ.all.Realize (Function.comp (⇑f) Inhabited.default) (Function.comp (⇑f …
    -/
    simp only [BoundedFormula.realize_all]
    /-
      case refine_5
      L : FirstOrder.Language
      M : Type u_1
      N : Type u_2
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f : L.Embedding M N
      htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
      n✝ : Nat
      φ✝ : L.BoundedFormula Empty n✝
      n : Nat
      φ : L.BoundedFormula Empty (HAdd.hAdd n 1)
      ih : ∀ (xs : Fin (HAdd.hAdd n 1) → M), Iff (φ.Realize (Function.comp (⇑f) Inha …
      xs : Fin n → M
      ⊢ Iff (∀ (a : N), φ.Realize (Function.comp (⇑f) Inhabited.default) (Fin.snoc ( …
    -/
    refine ⟨fun h a => ?_, ?_⟩
      /-
        case refine_5.refine_1
        L : FirstOrder.Language
        M : Type u_1
        N : Type u_2
        inst✝¹ : L.Structure M
        inst✝ : L.Structure N
        f : L.Embedding M N
        htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
        n✝ : Nat
        φ✝ : L.BoundedFormula Empty n✝
        n : Nat
        φ : L.BoundedFormula Empty (HAdd.hAdd n 1)
        ih : ∀ (xs : Fin (HAdd.hAdd n 1) → M), Iff (φ.Realize (Function.comp (⇑f) Inha …
        xs : Fin n → M
        h : ∀ (a : N), φ.Realize (Function.comp (⇑f) Inhabited.default) (Fin.snoc (Fun …
        a : M
        ⊢ φ.Realize Inhabited.default (Fin.snoc xs a)
      -/
    · rw [← ih, Fin.comp_snoc]
      /-
        case refine_5.refine_1
        L : FirstOrder.Language
        M : Type u_1
        N : Type u_2
        inst✝¹ : L.Structure M
        inst✝ : L.Structure N
        f : L.Embedding M N
        htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
        n✝ : Nat
        φ✝ : L.BoundedFormula Empty n✝
        n : Nat
        φ : L.BoundedFormula Empty (HAdd.hAdd n 1)
        ih : ∀ (xs : Fin (HAdd.hAdd n 1) → M), Iff (φ.Realize (Function.comp (⇑f) Inha …
        xs : Fin n → M
        h : ∀ (a : N), φ.Realize (Function.comp (⇑f) Inhabited.default) (Fin.snoc (Fun …
        a : M
        ⊢ φ.Realize (Function.comp (⇑f) Inhabited.default) (Fin.snoc (Function.comp (⇑ …
      -/
      exact h (f a)
      /-
        🎉 no goals
      -/
      /-
        case refine_5.refine_2
        L : FirstOrder.Language
        M : Type u_1
        N : Type u_2
        inst✝¹ : L.Structure M
        inst✝ : L.Structure N
        f : L.Embedding M N
        htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
        n✝ : Nat
        φ✝ : L.BoundedFormula Empty n✝
        n : Nat
        φ : L.BoundedFormula Empty (HAdd.hAdd n 1)
        ih : ∀ (xs : Fin (HAdd.hAdd n 1) → M), Iff (φ.Realize (Function.comp (⇑f) Inha …
        xs : Fin n → M
        ⊢ (∀ (a : M), φ.Realize Inhabited.default (Fin.snoc xs a)) → ∀ (a : N), φ.Real …
      -/
    · contrapose!
      /-
        case refine_5.refine_2
        L : FirstOrder.Language
        M : Type u_1
        N : Type u_2
        inst✝¹ : L.Structure M
        inst✝ : L.Structure N
        f : L.Embedding M N
        htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
        n✝ : Nat
        φ✝ : L.BoundedFormula Empty n✝
        n : Nat
        φ : L.BoundedFormula Empty (HAdd.hAdd n 1)
        ih : ∀ (xs : Fin (HAdd.hAdd n 1) → M), Iff (φ.Realize (Function.comp (⇑f) Inha …
        xs : Fin n → M
        ⊢ (Exists fun a => Not (φ.Realize (Function.comp (⇑f) Inhabited.default) (Fin. …
      -/
      rintro ⟨a, ha⟩
      obtain ⟨b, hb⟩ := htv n φ.not xs a (by
          rw [BoundedFormula.realize_not, ← Unique.eq_default (f ∘ default)]
          exact ha)
      /-
        case refine_5.refine_2.intro.intro
        L : FirstOrder.Language
        M : Type u_1
        N : Type u_2
        inst✝¹ : L.Structure M
        inst✝ : L.Structure N
        f : L.Embedding M N
        htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
        n✝ : Nat
        φ✝ : L.BoundedFormula Empty n✝
        n : Nat
        φ : L.BoundedFormula Empty (HAdd.hAdd n 1)
        ih : ∀ (xs : Fin (HAdd.hAdd n 1) → M), Iff (φ.Realize (Function.comp (⇑f) Inha …
        xs : Fin n → M
        a : N
        ha : Not (φ.Realize (Function.comp (⇑f) Inhabited.default) (Fin.snoc (Function …
        b : M
        hb : φ.not.Realize Inhabited.default (Fin.snoc (Function.comp (⇑f) xs) (f b))
        ⊢ Exists fun a => Not (φ.Realize Inhabited.default (Fin.snoc xs a))
      -/
      refine ⟨b, fun h => hb (Eq.mp ?_ ((ih _).2 h))⟩
      /-
        case refine_5.refine_2.intro.intro
        L : FirstOrder.Language
        M : Type u_1
        N : Type u_2
        inst✝¹ : L.Structure M
        inst✝ : L.Structure N
        f : L.Embedding M N
        htv : ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → M) …
        n✝ : Nat
        φ✝ : L.BoundedFormula Empty n✝
        n : Nat
        φ : L.BoundedFormula Empty (HAdd.hAdd n 1)
        ih : ∀ (xs : Fin (HAdd.hAdd n 1) → M), Iff (φ.Realize (Function.comp (⇑f) Inha …
        xs : Fin n → M
        a : N
        ha : Not (φ.Realize (Function.comp (⇑f) Inhabited.default) (Fin.snoc (Function …
        b : M
        hb : φ.not.Realize Inhabited.default (Fin.snoc (Function.comp (⇑f) xs) (f b))
        h : φ.Realize Inhabited.default (Fin.snoc xs b)
        ⊢ Eq (φ.Realize (Function.comp (⇑f) Inhabited.default) (Function.comp (⇑f) (Fi …
      -/
      rw [Unique.eq_default (f ∘ default), Fin.comp_snoc]
      /-
        🎉 no goals
      -/


/-- Bundles an embedding satisfying the Tarski-Vaught test as an elementary embedding. -/
@[simps]
def toElementaryEmbedding (f : M ↪[L] N)
    (htv :
      ∀ (n : ℕ) (φ : L.BoundedFormula Empty (n + 1)) (x : Fin n → M) (a : N),
        φ.Realize default (Fin.snoc (f ∘ x) a : _ → N) →
          ∃ b : M, φ.Realize default (Fin.snoc (f ∘ x) (f b) : _ → N)) :
    M ↪ₑ[L] N :=
  ⟨f, fun _ => f.isElementary_of_exists htv⟩


/-- A first-order equivalence is also an elementary embedding. -/
def toElementaryEmbedding (f : M ≃[L] N) : M ↪ₑ[L] N where
  toFun := f
                           /-
                             L : FirstOrder.Language
                             M : Type u_1
                             N : Type u_2
                             P : Type u_3
                             Q : Type u_4
                             inst✝³ : L.Structure M
                             inst✝² : L.Structure N
                             inst✝¹ : L.Structure P
                             inst✝ : L.Structure Q
                             f : L.Equiv M N
                             n : Nat
                             φ : L.Formula (Fin n)
                             x : Fin n → M
                             ⊢ Iff (φ.Realize (Function.comp (⇑f) x)) (φ.Realize x)
                           -/
  map_formula' n φ x := by aesop
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem toElementaryEmbedding_toEmbedding (f : M ≃[L] N) :
    f.toElementaryEmbedding.toEmbedding = f.toEmbedding :=
  rfl


@[simp]
theorem coe_toElementaryEmbedding (f : M ≃[L] N) :
    (f.toElementaryEmbedding : M → N) = (f : M → N) :=
  rfl


@[simp]
theorem realize_term_substructure {α : Type*} {S : L.Substructure M} (v : α → S) (t : L.Term α) :
    t.realize ((↑) ∘ v) = (↑(t.realize v) : M) :=
  HomClass.realize_term S.subtype


