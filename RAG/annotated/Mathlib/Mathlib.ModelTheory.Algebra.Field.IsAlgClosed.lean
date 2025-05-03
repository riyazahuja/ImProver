/-- A generic monic polynomial of degree `n` as an element of the
free commutative ring in `n+1` variables, with a variable for each
of the `n` non-leading coefficients of the polynomial and one variable (`Fin.last n`)
for `X`.  -/
def genericMonicPoly (n : ℕ) : FreeCommRing (Fin (n + 1)) :=
  of (Fin.last _) ^ n + ∑ i : Fin n, of i.castSucc * of (Fin.last _) ^ (i : ℕ)


theorem lift_genericMonicPoly [CommRing K] [Nontrivial K] {n : ℕ} (v : Fin (n+1) → K) :
    FreeCommRing.lift v (genericMonicPoly n) =
    (((monicEquivDegreeLT n).trans (degreeLTEquiv K n).toEquiv).symm (v ∘ Fin.castSucc)).1.eval
      (v (Fin.last _)) := by
  simp only [genericMonicPoly, map_add, map_pow, lift_of, map_sum, map_mul, monicEquivDegreeLT,
    degreeLTEquiv, Equiv.symm_trans_apply, LinearEquiv.coe_toEquiv_symm, EquivLike.coe_coe,
    LinearEquiv.coe_symm_mk, Function.comp_apply, Equiv.coe_fn_symm_mk, eval_add, eval_pow, eval_X,
    eval_finset_sum, eval_monomial]


/-- A sentence saying every monic polynomial of degree `n` has a root. -/
noncomputable def genericMonicPolyHasRoot (n : ℕ) : Language.ring.Sentence :=
  (∃' ((termOfFreeCommRing (genericMonicPoly n)).relabel Sum.inr =' 0)).alls


theorem realize_genericMonicPolyHasRoot [Field K] [CompatibleRing K] (n : ℕ) :
    K ⊨ genericMonicPolyHasRoot n ↔
      ∀ p : { p : K[X] // p.Monic ∧ p.natDegree = n }, ∃ x, p.1.eval x = 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    n : Nat
    ⊢ Iff (FirstOrder.Language.Sentence.Realize K (FirstOrder.Field.genericMonicPo …
  -/
  let _ := Classical.decEq K
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    n : Nat
    x✝ : DecidableEq K := Classical.decEq K
    ⊢ Iff (FirstOrder.Language.Sentence.Realize K (FirstOrder.Field.genericMonicPo …
  -/
  rw [Equiv.forall_congr_left ((monicEquivDegreeLT n).trans (degreeLTEquiv K n).toEquiv)]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    n : Nat
    x✝ : DecidableEq K := Classical.decEq K
    ⊢ Iff (FirstOrder.Language.Sentence.Realize K (FirstOrder.Field.genericMonicPo …
  -/
  simp [Sentence.Realize, genericMonicPolyHasRoot, lift_genericMonicPoly]
  /-
    🎉 no goals
  -/


/-- The theory of algebraically closed fields of characteristic `p` as a theory over
the language of rings -/
def _root_.FirstOrder.Language.Theory.ACF (p : ℕ) : Theory .ring :=
  Theory.fieldOfChar p ∪ genericMonicPolyHasRoot '' {n | 0 < n}


instance [Language.ring.Structure K] (p : ℕ) [h : (Theory.ACF p).Model K] :
    (Theory.fieldOfChar p).Model K :=
  Theory.Model.mono h Set.subset_union_left


instance [Field K] [CompatibleRing K] {p : ℕ} [CharP K p] [IsAlgClosed K] :
    (Theory.ACF p).Model K := by
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : FirstOrder.Ring.CompatibleRing K
    p : Nat
    inst✝¹ : CharP K p
    inst✝ : IsAlgClosed K
    ⊢ FirstOrder.Language.Theory.Model K (FirstOrder.Language.Theory.ACF p)
  -/
  refine Theory.model_union_iff.2 ⟨inferInstance, ?_⟩
  simp only [Theory.model_iff, Set.mem_image, Set.mem_singleton_iff,
    exists_prop, forall_exists_index, and_imp]
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : FirstOrder.Ring.CompatibleRing K
    p : Nat
    inst✝¹ : CharP K p
    inst✝ : IsAlgClosed K
    ⊢ ∀ (φ : FirstOrder.Language.ring.Sentence) (x : Nat), Membership.mem (setOf f …
  -/
  rintro _ n hn0 rfl
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : FirstOrder.Ring.CompatibleRing K
    p : Nat
    inst✝¹ : CharP K p
    inst✝ : IsAlgClosed K
    n : Nat
    hn0 : Membership.mem (setOf fun n => LT.lt 0 n) n
    ⊢ FirstOrder.Language.Sentence.Realize K (FirstOrder.Field.genericMonicPolyHas …
  -/
  simp only [realize_genericMonicPolyHasRoot]
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : FirstOrder.Ring.CompatibleRing K
    p : Nat
    inst✝¹ : CharP K p
    inst✝ : IsAlgClosed K
    n : Nat
    hn0 : Membership.mem (setOf fun n => LT.lt 0 n) n
    ⊢ ∀ (p : Subtype fun p => And p.Monic (Eq p.natDegree n)), Exists fun x => Eq  …
  -/
  rintro ⟨p, _, rfl⟩
  exact IsAlgClosed.exists_root p (ne_of_gt
    (natDegree_pos_iff_degree_pos.1 hn0))


theorem modelField_of_modelACF (p : ℕ) (K : Type*) [Language.ring.Structure K]
    [h : (Theory.ACF p).Model K] : Theory.field.Model K :=
  Theory.Model.mono h (Set.subset_union_of_subset_left Set.subset_union_left _)


/-- A model for the Theory of algebraically closed fields is a Field. After introducing
this as a local instance on a particular Type, you should usually also introduce
`modelField_of_modelACF p M`, `compatibleRingOfModelField` and `isAlgClosed_of_model_ACF` -/
@[reducible]
noncomputable def fieldOfModelACF (p : ℕ) (K : Type*)
    [Language.ring.Structure K]
    [h : (Theory.ACF p).Model K] : Field K := by
  /-
    K✝ : Type u_1
    p : Nat
    K : Type u_2
    inst✝ : FirstOrder.Language.ring.Structure K
    h : FirstOrder.Language.Theory.Model K (FirstOrder.Language.Theory.ACF p)
    ⊢ Field K
  -/
  have := modelField_of_modelACF p K
  /-
    K✝ : Type u_1
    p : Nat
    K : Type u_2
    inst✝ : FirstOrder.Language.ring.Structure K
    h : FirstOrder.Language.Theory.Model K (FirstOrder.Language.Theory.ACF p)
    this : FirstOrder.Language.Theory.Model K FirstOrder.Language.Theory.field
    ⊢ Field K
  -/
  exact fieldOfModelField K
  /-
    🎉 no goals
  -/


theorem isAlgClosed_of_model_ACF (p : ℕ) (K : Type*)
    [Field K] [CompatibleRing K] [h : (Theory.ACF p).Model K] :
    IsAlgClosed K := by
  /-
    p : Nat
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    h : FirstOrder.Language.Theory.Model K (FirstOrder.Language.Theory.ACF p)
    ⊢ IsAlgClosed K
  -/
  refine IsAlgClosed.of_exists_root _ ?_
  /-
    p : Nat
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    h : FirstOrder.Language.Theory.Model K (FirstOrder.Language.Theory.ACF p)
    ⊢ ∀ (p : Polynomial K), p.Monic → Irreducible p → Exists fun x => Eq (Polynomi …
  -/
  intro p hpm hpi
  have h : K ⊨ genericMonicPolyHasRoot '' {n | 0 < n} :=
    Theory.Model.mono h (by simp [Theory.ACF])
  simp only [Theory.model_iff, Set.mem_image, Set.mem_singleton_iff,
    exists_prop, forall_exists_index, and_imp] at h
  have := h _ p.natDegree (natDegree_pos_iff_degree_pos.2
    (degree_pos_of_irreducible hpi)) rfl
  /-
    p✝ : Nat
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    h✝ : FirstOrder.Language.Theory.Model K (FirstOrder.Language.Theory.ACF p✝)
    p : Polynomial K
    hpm : p.Monic
    hpi : Irreducible p
    h : ∀ (φ : FirstOrder.Language.ring.Sentence) (x : Nat), Membership.mem (setOf …
    this : FirstOrder.Language.Sentence.Realize K (FirstOrder.Field.genericMonicPo …
    ⊢ Exists fun x => Eq (Polynomial.eval x p) 0
  -/
  rw [realize_genericMonicPolyHasRoot] at this
  /-
    p✝ : Nat
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    h✝ : FirstOrder.Language.Theory.Model K (FirstOrder.Language.Theory.ACF p✝)
    p : Polynomial K
    hpm : p.Monic
    hpi : Irreducible p
    h : ∀ (φ : FirstOrder.Language.ring.Sentence) (x : Nat), Membership.mem (setOf …
    this : ∀ (p_1 : Subtype fun p_1 => And p_1.Monic (Eq p_1.natDegree p.natDegree …
    ⊢ Exists fun x => Eq (Polynomial.eval x p) 0
  -/
  exact this ⟨_, hpm, rfl⟩
  /-
    🎉 no goals
  -/


theorem ACF_isSatisfiable {p : ℕ} (hp : p.Prime ∨ p = 0) :
    (Theory.ACF p).IsSatisfiable := by
  cases hp with
  | inl hp =>
    have : Fact p.Prime := ⟨hp⟩
    let _ := compatibleRingOfRing (AlgebraicClosure (ZMod p))
    have : CharP (AlgebraicClosure (ZMod p)) p :=
      charP_of_injective_algebraMap
        (RingHom.injective (algebraMap (ZMod p) (AlgebraicClosure (ZMod p)))) p
    exact ⟨⟨AlgebraicClosure (ZMod p)⟩⟩
  | inr hp =>
    subst hp
    let _ := compatibleRingOfRing (AlgebraicClosure ℚ)
    have : CharP (AlgebraicClosure ℚ) 0 :=
      charP_of_injective_algebraMap
        (RingHom.injective (algebraMap ℚ (AlgebraicClosure ℚ))) 0
    exact ⟨⟨AlgebraicClosure ℚ⟩⟩


/-- The Theory `Theory.ACF p` is `κ`-categorical whenever `κ` is an uncountable cardinal. -/
theorem ACF_categorical {p : ℕ} (κ : Cardinal) (hκ : ℵ₀ < κ) :
    Categorical κ (Theory.ACF p) := by
  /-
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    ⊢ κ.Categorical (FirstOrder.Language.Theory.ACF p)
  -/
  rintro ⟨M⟩ ⟨N⟩ hM hN
  /-
    case mk.mk
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    ⊢ Nonempty (FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelT …
  -/
  let _ := fieldOfModelACF p M
  /-
    case mk.mk
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝ : Field M := FirstOrder.Field.fieldOfModelACF p M
    ⊢ Nonempty (FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelT …
  -/
  have := modelField_of_modelACF p M
  /-
    case mk.mk
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝ : Field M := FirstOrder.Field.fieldOfModelACF p M
    this : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    ⊢ Nonempty (FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelT …
  -/
  let _ := compatibleRingOfModelField M
  /-
    case mk.mk
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝¹ : Field M := FirstOrder.Field.fieldOfModelACF p M
    this : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    x✝ : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMode …
    ⊢ Nonempty (FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelT …
  -/
  have := isAlgClosed_of_model_ACF p M
  /-
    case mk.mk
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝¹ : Field M := FirstOrder.Field.fieldOfModelACF p M
    this✝ : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    x✝ : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMode …
    this : IsAlgClosed M
    ⊢ Nonempty (FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelT …
  -/
  have := charP_of_model_fieldOfChar p M
  /-
    case mk.mk
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝¹ : Field M := FirstOrder.Field.fieldOfModelACF p M
    this✝¹ : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    x✝ : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMode …
    this✝ : IsAlgClosed M
    this : CharP M p
    ⊢ Nonempty (FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelT …
  -/
  let _ := fieldOfModelACF p N
  /-
    case mk.mk
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝² : Field M := FirstOrder.Field.fieldOfModelACF p M
    this✝¹ : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    x✝¹ : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMod …
    this✝ : IsAlgClosed M
    this : CharP M p
    x✝ : Field N := FirstOrder.Field.fieldOfModelACF p N
    ⊢ Nonempty (FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelT …
  -/
  have := modelField_of_modelACF p N
  /-
    case mk.mk
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝² : Field M := FirstOrder.Field.fieldOfModelACF p M
    this✝² : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    x✝¹ : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMod …
    this✝¹ : IsAlgClosed M
    this✝ : CharP M p
    x✝ : Field N := FirstOrder.Field.fieldOfModelACF p N
    this : FirstOrder.Language.Theory.Model N FirstOrder.Language.Theory.field
    ⊢ Nonempty (FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelT …
  -/
  let _ := compatibleRingOfModelField N
  /-
    case mk.mk
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝³ : Field M := FirstOrder.Field.fieldOfModelACF p M
    this✝² : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    x✝² : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMod …
    this✝¹ : IsAlgClosed M
    this✝ : CharP M p
    x✝¹ : Field N := FirstOrder.Field.fieldOfModelACF p N
    this : FirstOrder.Language.Theory.Model N FirstOrder.Language.Theory.field
    x✝ : FirstOrder.Ring.CompatibleRing N := FirstOrder.Field.compatibleRingOfMode …
    ⊢ Nonempty (FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelT …
  -/
  have := isAlgClosed_of_model_ACF p N
  /-
    case mk.mk
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝³ : Field M := FirstOrder.Field.fieldOfModelACF p M
    this✝³ : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    x✝² : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMod …
    this✝² : IsAlgClosed M
    this✝¹ : CharP M p
    x✝¹ : Field N := FirstOrder.Field.fieldOfModelACF p N
    this✝ : FirstOrder.Language.Theory.Model N FirstOrder.Language.Theory.field
    x✝ : FirstOrder.Ring.CompatibleRing N := FirstOrder.Field.compatibleRingOfMode …
    this : IsAlgClosed N
    ⊢ Nonempty (FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelT …
  -/
  have := charP_of_model_fieldOfChar p N
  /-
    case mk.mk
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝³ : Field M := FirstOrder.Field.fieldOfModelACF p M
    this✝⁴ : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    x✝² : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMod …
    this✝³ : IsAlgClosed M
    this✝² : CharP M p
    x✝¹ : Field N := FirstOrder.Field.fieldOfModelACF p N
    this✝¹ : FirstOrder.Language.Theory.Model N FirstOrder.Language.Theory.field
    x✝ : FirstOrder.Ring.CompatibleRing N := FirstOrder.Field.compatibleRingOfMode …
    this✝ : IsAlgClosed N
    this : CharP N p
    ⊢ Nonempty (FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelT …
  -/
  constructor
  /-
    case mk.mk.val
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝³ : Field M := FirstOrder.Field.fieldOfModelACF p M
    this✝⁴ : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    x✝² : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMod …
    this✝³ : IsAlgClosed M
    this✝² : CharP M p
    x✝¹ : Field N := FirstOrder.Field.fieldOfModelACF p N
    this✝¹ : FirstOrder.Language.Theory.Model N FirstOrder.Language.Theory.field
    x✝ : FirstOrder.Ring.CompatibleRing N := FirstOrder.Field.compatibleRingOfMode …
    this✝ : IsAlgClosed N
    this : CharP N p
    ⊢ FirstOrder.Language.ring.Equiv ↑(FirstOrder.Language.Theory.ModelType.mk M)  …
  -/
  refine languageEquivEquivRingEquiv.symm ?_
  /-
    case mk.mk.val
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝³ : Field M := FirstOrder.Field.fieldOfModelACF p M
    this✝⁴ : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    x✝² : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMod …
    this✝³ : IsAlgClosed M
    this✝² : CharP M p
    x✝¹ : Field N := FirstOrder.Field.fieldOfModelACF p N
    this✝¹ : FirstOrder.Language.Theory.Model N FirstOrder.Language.Theory.field
    x✝ : FirstOrder.Ring.CompatibleRing N := FirstOrder.Field.compatibleRingOfMode …
    this✝ : IsAlgClosed N
    this : CharP N p
    ⊢ RingEquiv ↑(FirstOrder.Language.Theory.ModelType.mk M) ↑(FirstOrder.Language …
  -/
  apply Classical.choice
  /-
    case mk.mk.val.a
    p : Nat
    κ : Cardinal.{u_2}
    hκ : LT.lt Cardinal.aleph0 κ
    M : Type u_2
    struc✝¹ : FirstOrder.Language.ring.Structure M
    is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
    nonempty'✝¹ : Nonempty M
    N : Type u_2
    struc✝ : FirstOrder.Language.ring.Structure N
    is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
    nonempty'✝ : Nonempty N
    hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
    hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
    x✝³ : Field M := FirstOrder.Field.fieldOfModelACF p M
    this✝⁴ : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
    x✝² : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMod …
    this✝³ : IsAlgClosed M
    this✝² : CharP M p
    x✝¹ : Field N := FirstOrder.Field.fieldOfModelACF p N
    this✝¹ : FirstOrder.Language.Theory.Model N FirstOrder.Language.Theory.field
    x✝ : FirstOrder.Ring.CompatibleRing N := FirstOrder.Field.compatibleRingOfMode …
    this✝ : IsAlgClosed N
    this : CharP N p
    ⊢ Nonempty (RingEquiv ↑(FirstOrder.Language.Theory.ModelType.mk M) ↑(FirstOrde …
  -/
  refine IsAlgClosed.ringEquiv_of_equiv_of_char_eq p ?_ ?_
    /-
      case mk.mk.val.a.refine_1
      p : Nat
      κ : Cardinal.{u_2}
      hκ : LT.lt Cardinal.aleph0 κ
      M : Type u_2
      struc✝¹ : FirstOrder.Language.ring.Structure M
      is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
      nonempty'✝¹ : Nonempty M
      N : Type u_2
      struc✝ : FirstOrder.Language.ring.Structure N
      is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
      nonempty'✝ : Nonempty N
      hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
      hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
      x✝³ : Field M := FirstOrder.Field.fieldOfModelACF p M
      this✝⁴ : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
      x✝² : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMod …
      this✝³ : IsAlgClosed M
      this✝² : CharP M p
      x✝¹ : Field N := FirstOrder.Field.fieldOfModelACF p N
      this✝¹ : FirstOrder.Language.Theory.Model N FirstOrder.Language.Theory.field
      x✝ : FirstOrder.Ring.CompatibleRing N := FirstOrder.Field.compatibleRingOfMode …
      this✝ : IsAlgClosed N
      this : CharP N p
      ⊢ LT.lt Cardinal.aleph0 (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk …
    -/
  · rw [hM]; exact hκ
             /-
               🎉 no goals
             -/
    /-
      case mk.mk.val.a.refine_2
      p : Nat
      κ : Cardinal.{u_2}
      hκ : LT.lt Cardinal.aleph0 κ
      M : Type u_2
      struc✝¹ : FirstOrder.Language.ring.Structure M
      is_model✝¹ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.AC …
      nonempty'✝¹ : Nonempty M
      N : Type u_2
      struc✝ : FirstOrder.Language.ring.Structure N
      is_model✝ : FirstOrder.Language.Theory.Model N (FirstOrder.Language.Theory.ACF …
      nonempty'✝ : Nonempty N
      hM : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk M)) κ
      hN : Eq (Cardinal.mk ↑(FirstOrder.Language.Theory.ModelType.mk N)) κ
      x✝³ : Field M := FirstOrder.Field.fieldOfModelACF p M
      this✝⁴ : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
      x✝² : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMod …
      this✝³ : IsAlgClosed M
      this✝² : CharP M p
      x✝¹ : Field N := FirstOrder.Field.fieldOfModelACF p N
      this✝¹ : FirstOrder.Language.Theory.Model N FirstOrder.Language.Theory.field
      x✝ : FirstOrder.Ring.CompatibleRing N := FirstOrder.Field.compatibleRingOfMode …
      this✝ : IsAlgClosed N
      this : CharP N p
      ⊢ Nonempty (Equiv ↑(FirstOrder.Language.Theory.ModelType.mk M) ↑(FirstOrder.La …
    -/
  · rw [← Cardinal.eq, hM, hN]
    /-
      🎉 no goals
    -/


theorem ACF_isComplete {p : ℕ} (hp : p.Prime ∨ p = 0) :
    (Theory.ACF p).IsComplete := by
  apply Categorical.isComplete.{0, 0, 0} (Order.succ ℵ₀) _
    (ACF_categorical _ (Order.lt_succ _))
    (Order.le_succ ℵ₀)
    /-
      case h2
      p : Nat
      hp : Or (Nat.Prime p) (Eq p 0)
      ⊢ LE.le (Cardinal.lift.{0, 0} FirstOrder.Language.ring.card) (Cardinal.lift.{0 …
    -/
  · simp only [card_ring, lift_id']
    /-
      case h2
      p : Nat
      hp : Or (Nat.Prime p) (Eq p 0)
      ⊢ LE.le 5 (Order.succ Cardinal.aleph0)
    -/
    exact le_trans (le_of_lt (lt_aleph0_of_finite _)) (Order.le_succ _)
    /-
      🎉 no goals
    -/
    /-
      case hS
      p : Nat
      hp : Or (Nat.Prime p) (Eq p 0)
      ⊢ (FirstOrder.Language.Theory.ACF p).IsSatisfiable
    -/
  · exact ACF_isSatisfiable hp
    /-
      🎉 no goals
    -/
    /-
      case hT
      p : Nat
      hp : Or (Nat.Prime p) (Eq p 0)
      ⊢ ∀ (M : (FirstOrder.Language.Theory.ACF p).ModelType), Infinite ↑M
    -/
  · rintro ⟨M⟩
    /-
      case hT.mk
      p : Nat
      hp : Or (Nat.Prime p) (Eq p 0)
      M : Type
      struc✝ : FirstOrder.Language.ring.Structure M
      is_model✝ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.ACF …
      nonempty'✝ : Nonempty M
      ⊢ Infinite ↑(FirstOrder.Language.Theory.ModelType.mk M)
    -/
    let _ := fieldOfModelACF p M
    /-
      case hT.mk
      p : Nat
      hp : Or (Nat.Prime p) (Eq p 0)
      M : Type
      struc✝ : FirstOrder.Language.ring.Structure M
      is_model✝ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.ACF …
      nonempty'✝ : Nonempty M
      x✝ : Field M := FirstOrder.Field.fieldOfModelACF p M
      ⊢ Infinite ↑(FirstOrder.Language.Theory.ModelType.mk M)
    -/
    have := modelField_of_modelACF p M
    /-
      case hT.mk
      p : Nat
      hp : Or (Nat.Prime p) (Eq p 0)
      M : Type
      struc✝ : FirstOrder.Language.ring.Structure M
      is_model✝ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.ACF …
      nonempty'✝ : Nonempty M
      x✝ : Field M := FirstOrder.Field.fieldOfModelACF p M
      this : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
      ⊢ Infinite ↑(FirstOrder.Language.Theory.ModelType.mk M)
    -/
    let _ := compatibleRingOfModelField M
    /-
      case hT.mk
      p : Nat
      hp : Or (Nat.Prime p) (Eq p 0)
      M : Type
      struc✝ : FirstOrder.Language.ring.Structure M
      is_model✝ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.ACF …
      nonempty'✝ : Nonempty M
      x✝¹ : Field M := FirstOrder.Field.fieldOfModelACF p M
      this : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
      x✝ : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMode …
      ⊢ Infinite ↑(FirstOrder.Language.Theory.ModelType.mk M)
    -/
    have := isAlgClosed_of_model_ACF p M
    /-
      case hT.mk
      p : Nat
      hp : Or (Nat.Prime p) (Eq p 0)
      M : Type
      struc✝ : FirstOrder.Language.ring.Structure M
      is_model✝ : FirstOrder.Language.Theory.Model M (FirstOrder.Language.Theory.ACF …
      nonempty'✝ : Nonempty M
      x✝¹ : Field M := FirstOrder.Field.fieldOfModelACF p M
      this✝ : FirstOrder.Language.Theory.Model M FirstOrder.Language.Theory.field
      x✝ : FirstOrder.Ring.CompatibleRing M := FirstOrder.Field.compatibleRingOfMode …
      this : IsAlgClosed M
      ⊢ Infinite ↑(FirstOrder.Language.Theory.ModelType.mk M)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


theorem finite_ACF_prime_not_realize_of_ACF_zero_realize
    (φ : Language.ring.Sentence) (h : Theory.ACF 0 ⊨ᵇ φ) :
    Set.Finite { p : Nat.Primes | ¬ Theory.ACF p ⊨ᵇ φ } := by
  /-
    φ : FirstOrder.Language.ring.Sentence
    h : (FirstOrder.Language.Theory.ACF 0).ModelsBoundedFormula φ
    ⊢ (setOf fun p => Not ((FirstOrder.Language.Theory.ACF ↑p).ModelsBoundedFormul …
  -/
  rw [Theory.models_iff_finset_models] at h
  /-
    φ : FirstOrder.Language.ring.Sentence
    h : Exists fun T0 => And (HasSubset.Subset (↑T0) (FirstOrder.Language.Theory.A …
    ⊢ (setOf fun p => Not ((FirstOrder.Language.Theory.ACF ↑p).ModelsBoundedFormul …
  -/
  rcases h with ⟨T0, hT0, h⟩
  have f : ∀ ψ ∈ Theory.ACF 0,
      { s : Finset Nat.Primes // ∀ q : Nat.Primes, q ∉ s → Theory.ACF q ⊨ᵇ ψ } := by
    intro ψ hψ
    rw [Theory.ACF, Theory.fieldOfChar, Set.union_right_comm, Set.mem_union, if_pos rfl,
      Set.mem_image] at hψ
    apply Classical.choice
    rcases hψ with h | ⟨p, hp, rfl⟩
    · refine ⟨⟨∅, ?_⟩⟩
      intro q _
      exact Theory.models_sentence_of_mem
        (by rw [Theory.ACF, Theory.fieldOfChar, Set.union_right_comm];
            exact Set.mem_union_left _ h)
    · refine ⟨⟨{⟨p, hp⟩}, ?_⟩⟩
      rintro ⟨q, _⟩ hq ⟨K⟩ _ _
      have hqp : q ≠ p := by simpa [← Nat.Primes.coe_nat_inj] using hq
      let _ := fieldOfModelACF q K
      have := modelField_of_modelACF q K
      let _ := compatibleRingOfModelField K
      have := charP_of_model_fieldOfChar q K
      simp only [eqZero, Term.equal, BoundedFormula.realize_not, BoundedFormula.realize_bdEqual,
        Term.realize_relabel, Sum.elim_comp_inl, realize_termOfFreeCommRing, map_natCast,
        realize_zero, ← CharP.charP_iff_prime_eq_zero hp]
      intro _
      exact hqp <| CharP.eq K inferInstance inferInstance
  /-
    case intro.intro
    φ : FirstOrder.Language.ring.Sentence
    T0 : Finset FirstOrder.Language.ring.Sentence
    hT0 : HasSubset.Subset (↑T0) (FirstOrder.Language.Theory.ACF 0)
    h : FirstOrder.Language.Theory.ModelsBoundedFormula (↑T0) φ
    f : (ψ : FirstOrder.Language.ring.Sentence) → Membership.mem (FirstOrder.Langu …
    ⊢ (setOf fun p => Not ((FirstOrder.Language.Theory.ACF ↑p).ModelsBoundedFormul …
  -/
  let s : Finset Nat.Primes := T0.attach.biUnion (fun φ => f φ.1 (hT0 φ.2))
  have hs : ∀ (p : Nat.Primes) ψ, ψ ∈ T0 → p ∉ s → Theory.ACF p ⊨ᵇ ψ := by
    intro p ψ hψ hpψ
    simp only [s, Finset.mem_biUnion, Finset.mem_attach, true_and,
      Subtype.exists, not_exists] at hpψ
    exact (f ψ (hT0 hψ)).2 p (hpψ _ hψ)
  /-
    case intro.intro
    φ : FirstOrder.Language.ring.Sentence
    T0 : Finset FirstOrder.Language.ring.Sentence
    hT0 : HasSubset.Subset (↑T0) (FirstOrder.Language.Theory.ACF 0)
    h : FirstOrder.Language.Theory.ModelsBoundedFormula (↑T0) φ
    f : (ψ : FirstOrder.Language.ring.Sentence) → Membership.mem (FirstOrder.Langu …
    s : Finset Nat.Primes := T0.attach.biUnion fun φ => ↑(f ↑φ ⋯)
    hs : ∀ (p : Nat.Primes) (ψ : FirstOrder.Language.ring.Sentence), Membership.me …
    ⊢ (setOf fun p => Not ((FirstOrder.Language.Theory.ACF ↑p).ModelsBoundedFormul …
  -/
  refine Set.Finite.subset (Finset.finite_toSet s) (Set.compl_subset_comm.2 ?_)
  /-
    case intro.intro
    φ : FirstOrder.Language.ring.Sentence
    T0 : Finset FirstOrder.Language.ring.Sentence
    hT0 : HasSubset.Subset (↑T0) (FirstOrder.Language.Theory.ACF 0)
    h : FirstOrder.Language.Theory.ModelsBoundedFormula (↑T0) φ
    f : (ψ : FirstOrder.Language.ring.Sentence) → Membership.mem (FirstOrder.Langu …
    s : Finset Nat.Primes := T0.attach.biUnion fun φ => ↑(f ↑φ ⋯)
    hs : ∀ (p : Nat.Primes) (ψ : FirstOrder.Language.ring.Sentence), Membership.me …
    ⊢ HasSubset.Subset (HasCompl.compl ↑s) fun p => ∀ (M : (FirstOrder.Language.Th …
  -/
  intro p hp
  /-
    case intro.intro
    φ : FirstOrder.Language.ring.Sentence
    T0 : Finset FirstOrder.Language.ring.Sentence
    hT0 : HasSubset.Subset (↑T0) (FirstOrder.Language.Theory.ACF 0)
    h : FirstOrder.Language.Theory.ModelsBoundedFormula (↑T0) φ
    f : (ψ : FirstOrder.Language.ring.Sentence) → Membership.mem (FirstOrder.Langu …
    s : Finset Nat.Primes := T0.attach.biUnion fun φ => ↑(f ↑φ ⋯)
    hs : ∀ (p : Nat.Primes) (ψ : FirstOrder.Language.ring.Sentence), Membership.me …
    p : Nat.Primes
    hp : Membership.mem (HasCompl.compl ↑s) p
    ⊢ Membership.mem (fun p => ∀ (M : (FirstOrder.Language.Theory.ACF ↑p).ModelTyp …
  -/
  exact Theory.models_of_models_theory (fun ψ hψ => hs p ψ hψ hp) h
  /-
    🎉 no goals
  -/


/-- The **Lefschetz principle**. A first order sentence is modeled by the theory
of algebraically closed fields of characteristic zero if and only if it is modeled by
the theory of algebraically closed fields of characteristic `p` for infinitely many `p`. -/
theorem ACF_zero_realize_iff_infinite_ACF_prime_realize {φ : Language.ring.Sentence} :
    Theory.ACF 0 ⊨ᵇ φ ↔ Set.Infinite { p : Nat.Primes | Theory.ACF p ⊨ᵇ φ } := by
  refine ⟨fun h => Set.infinite_of_finite_compl
      (finite_ACF_prime_not_realize_of_ACF_zero_realize φ h),
    not_imp_not.1 ?_⟩
  simpa [(ACF_isComplete (Or.inr rfl)).models_not_iff,
      fun p : Nat.Primes => (ACF_isComplete (Or.inl p.2)).models_not_iff] using
    finite_ACF_prime_not_realize_of_ACF_zero_realize φ.not


/-- Another statement of the **Lefschetz principle**. A first order sentence is modeled by the
theory of algebraically closed fields of characteristic zero if and only if it is modeled by the
theory of algebraically closed fields of characteristic `p` for all but finitely many primes `p`.
-/
theorem ACF_zero_realize_iff_finite_ACF_prime_not_realize {φ : Language.ring.Sentence} :
    Theory.ACF 0 ⊨ᵇ φ ↔ Set.Finite { p : Nat.Primes | Theory.ACF p ⊨ᵇ φ }ᶜ :=
  ⟨fun h => finite_ACF_prime_not_realize_of_ACF_zero_realize φ h,
    fun h => ACF_zero_realize_iff_infinite_ACF_prime_realize.2
      (Set.infinite_of_finite_compl h)⟩



