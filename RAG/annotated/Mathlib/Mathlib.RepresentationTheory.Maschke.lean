/-- We define the conjugate of `π` by `g`, as a `k`-linear map. -/
def conjugate (g : G) : W →ₗ[k] V :=
  GroupSMul.linearMap k V g⁻¹ ∘ₗ π ∘ₗ GroupSMul.linearMap k W g


theorem conjugate_apply (g : G) (v : W) :
    π.conjugate g v = MonoidAlgebra.single g⁻¹ (1 : k) • π (MonoidAlgebra.single g (1 : k) • v) :=
  rfl


theorem conjugate_i (h : ∀ v : V, π (i v) = v) (g : G) (v : V) :
    (conjugate π g : W → V) (i v) = v := by
  rw [conjugate_apply, ← i.map_smul, h, ← mul_smul, single_mul_single, mul_one, inv_mul_cancel,
    ← one_def, one_smul]


/-- The sum of the conjugates of `π` by each element `g : G`, as a `k`-linear map.

(We postpone dividing by the size of the group as long as possible.)
-/
def sumOfConjugates : W →ₗ[k] V :=
  ∑ g : G, π.conjugate g


lemma sumOfConjugates_apply (v : W) : π.sumOfConjugates G v = ∑ g : G, π.conjugate g v :=
  LinearMap.sum_apply _ _ _


/-- In fact, the sum over `g : G` of the conjugate of `π` by `g` is a `k[G]`-linear map.
-/
def sumOfConjugatesEquivariant : W →ₗ[MonoidAlgebra k G] V :=
  MonoidAlgebra.equivariantOfLinearOfComm (π.sumOfConjugates G) fun g v => by
    /-
      k : Type u
      inst✝¹⁰ : CommRing k
      G : Type u
      inst✝⁹ : Group G
      V : Type v
      inst✝⁸ : AddCommGroup V
      inst✝⁷ : Module k V
      inst✝⁶ : Module (MonoidAlgebra k G) V
      inst✝⁵ : IsScalarTower k (MonoidAlgebra k G) V
      W : Type w
      inst✝⁴ : AddCommGroup W
      inst✝³ : Module k W
      inst✝² : Module (MonoidAlgebra k G) W
      inst✝¹ : IsScalarTower k (MonoidAlgebra k G) W
      π : LinearMap (RingHom.id k) W V
      i : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
      inst✝ : Fintype G
      g : G
      v : W
      ⊢ Eq ((LinearMap.sumOfConjugates G π) (HSMul.hSMul (MonoidAlgebra.single g 1)  …
    -/
    simp only [sumOfConjugates_apply, Finset.smul_sum, conjugate_apply]
    /-
      k : Type u
      inst✝¹⁰ : CommRing k
      G : Type u
      inst✝⁹ : Group G
      V : Type v
      inst✝⁸ : AddCommGroup V
      inst✝⁷ : Module k V
      inst✝⁶ : Module (MonoidAlgebra k G) V
      inst✝⁵ : IsScalarTower k (MonoidAlgebra k G) V
      W : Type w
      inst✝⁴ : AddCommGroup W
      inst✝³ : Module k W
      inst✝² : Module (MonoidAlgebra k G) W
      inst✝¹ : IsScalarTower k (MonoidAlgebra k G) W
      π : LinearMap (RingHom.id k) W V
      i : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
      inst✝ : Fintype G
      g : G
      v : W
      ⊢ Eq (Finset.univ.sum fun x => HSMul.hSMul (MonoidAlgebra.single (Inv.inv x) 1 …
    -/
    refine Fintype.sum_bijective (· * g) (Group.mulRight_bijective g) _ _ fun i ↦ ?_
    /-
      k : Type u
      inst✝¹⁰ : CommRing k
      G : Type u
      inst✝⁹ : Group G
      V : Type v
      inst✝⁸ : AddCommGroup V
      inst✝⁷ : Module k V
      inst✝⁶ : Module (MonoidAlgebra k G) V
      inst✝⁵ : IsScalarTower k (MonoidAlgebra k G) V
      W : Type w
      inst✝⁴ : AddCommGroup W
      inst✝³ : Module k W
      inst✝² : Module (MonoidAlgebra k G) W
      inst✝¹ : IsScalarTower k (MonoidAlgebra k G) W
      π : LinearMap (RingHom.id k) W V
      i✝ : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
      inst✝ : Fintype G
      g : G
      v : W
      i : G
      ⊢ Eq (HSMul.hSMul (MonoidAlgebra.single (Inv.inv i) 1) (π (HSMul.hSMul (Monoid …
    -/
    simp only [smul_smul, single_mul_single, mul_inv_rev, mul_inv_cancel_left, one_mul]
    /-
      🎉 no goals
    -/


theorem sumOfConjugatesEquivariant_apply (v : W) :
    π.sumOfConjugatesEquivariant G v = ∑ g : G, π.conjugate g v :=
  π.sumOfConjugates_apply G v


/-- We construct our `k[G]`-linear retraction of `i` as
$$ \frac{1}{|G|} \sum_{g \in G} g⁻¹ • π(g • -). $$
-/
def equivariantProjection : W →ₗ[MonoidAlgebra k G] V :=
  Ring.inverse (Fintype.card G : k) • π.sumOfConjugatesEquivariant G


theorem equivariantProjection_apply (v : W) :
    π.equivariantProjection G v = Ring.inverse (Fintype.card G : k) • ∑ g : G, π.conjugate g v := by
  /-
    k : Type u
    inst✝¹⁰ : CommRing k
    G : Type u
    inst✝⁹ : Group G
    V : Type v
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module k V
    inst✝⁶ : Module (MonoidAlgebra k G) V
    inst✝⁵ : IsScalarTower k (MonoidAlgebra k G) V
    W : Type w
    inst✝⁴ : AddCommGroup W
    inst✝³ : Module k W
    inst✝² : Module (MonoidAlgebra k G) W
    inst✝¹ : IsScalarTower k (MonoidAlgebra k G) W
    π : LinearMap (RingHom.id k) W V
    inst✝ : Fintype G
    v : W
    ⊢ Eq ((LinearMap.equivariantProjection G π) v) (HSMul.hSMul (Ring.inverse ↑(Fi …
  -/
  simp only [equivariantProjection, smul_apply, sumOfConjugatesEquivariant_apply]
  /-
    🎉 no goals
  -/


theorem equivariantProjection_condition (hcard : IsUnit (Fintype.card G : k))
    (h : ∀ v : V, π (i v) = v) (v : V) : (π.equivariantProjection G) (i v) = v := by
  /-
    k : Type u
    inst✝¹⁰ : CommRing k
    G : Type u
    inst✝⁹ : Group G
    V : Type v
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module k V
    inst✝⁶ : Module (MonoidAlgebra k G) V
    inst✝⁵ : IsScalarTower k (MonoidAlgebra k G) V
    W : Type w
    inst✝⁴ : AddCommGroup W
    inst✝³ : Module k W
    inst✝² : Module (MonoidAlgebra k G) W
    inst✝¹ : IsScalarTower k (MonoidAlgebra k G) W
    π : LinearMap (RingHom.id k) W V
    i : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
    inst✝ : Fintype G
    hcard : IsUnit ↑(Fintype.card G)
    h : ∀ (v : V), Eq (π (i v)) v
    v : V
    ⊢ Eq ((LinearMap.equivariantProjection G π) (i v)) v
  -/
  rw [equivariantProjection_apply]
  /-
    k : Type u
    inst✝¹⁰ : CommRing k
    G : Type u
    inst✝⁹ : Group G
    V : Type v
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module k V
    inst✝⁶ : Module (MonoidAlgebra k G) V
    inst✝⁵ : IsScalarTower k (MonoidAlgebra k G) V
    W : Type w
    inst✝⁴ : AddCommGroup W
    inst✝³ : Module k W
    inst✝² : Module (MonoidAlgebra k G) W
    inst✝¹ : IsScalarTower k (MonoidAlgebra k G) W
    π : LinearMap (RingHom.id k) W V
    i : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
    inst✝ : Fintype G
    hcard : IsUnit ↑(Fintype.card G)
    h : ∀ (v : V), Eq (π (i v)) v
    v : V
    ⊢ Eq (HSMul.hSMul (Ring.inverse ↑(Fintype.card G)) (Finset.univ.sum fun g => ( …
  -/
  simp only [conjugate_i π i h]
  rw [Finset.sum_const, Finset.card_univ, ← Nat.cast_smul_eq_nsmul k, smul_smul,
    Ring.inverse_mul_cancel _ hcard, one_smul]


theorem exists_leftInverse_of_injective
    (f : V →ₗ[MonoidAlgebra k G] W) (hf : LinearMap.ker f = ⊥) :
    ∃ g : W →ₗ[MonoidAlgebra k G] V, g.comp f = LinearMap.id := by
  /-
    k : Type u
    inst✝⁷ : Field k
    G : Type u
    inst✝⁶ : Fintype G
    inst✝⁵ : NeZero ↑(Fintype.card G)
    inst✝⁴ : Group G
    V : Type u
    inst✝³ : AddCommGroup V
    inst✝² : Module (MonoidAlgebra k G) V
    W : Type u
    inst✝¹ : AddCommGroup W
    inst✝ : Module (MonoidAlgebra k G) W
    f : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
    hf : Eq (LinearMap.ker f) Bot.bot
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  let A := MonoidAlgebra k G
  /-
    k : Type u
    inst✝⁷ : Field k
    G : Type u
    inst✝⁶ : Fintype G
    inst✝⁵ : NeZero ↑(Fintype.card G)
    inst✝⁴ : Group G
    V : Type u
    inst✝³ : AddCommGroup V
    inst✝² : Module (MonoidAlgebra k G) V
    W : Type u
    inst✝¹ : AddCommGroup W
    inst✝ : Module (MonoidAlgebra k G) W
    f : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
    hf : Eq (LinearMap.ker f) Bot.bot
    A : Type u := MonoidAlgebra k G
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  letI : Module k W := .compHom W (algebraMap k A)
  /-
    k : Type u
    inst✝⁷ : Field k
    G : Type u
    inst✝⁶ : Fintype G
    inst✝⁵ : NeZero ↑(Fintype.card G)
    inst✝⁴ : Group G
    V : Type u
    inst✝³ : AddCommGroup V
    inst✝² : Module (MonoidAlgebra k G) V
    W : Type u
    inst✝¹ : AddCommGroup W
    inst✝ : Module (MonoidAlgebra k G) W
    f : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
    hf : Eq (LinearMap.ker f) Bot.bot
    A : Type u := MonoidAlgebra k G
    this : Module k W := Module.compHom W (algebraMap k A)
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  letI : Module k V := .compHom V (algebraMap k A)
  /-
    k : Type u
    inst✝⁷ : Field k
    G : Type u
    inst✝⁶ : Fintype G
    inst✝⁵ : NeZero ↑(Fintype.card G)
    inst✝⁴ : Group G
    V : Type u
    inst✝³ : AddCommGroup V
    inst✝² : Module (MonoidAlgebra k G) V
    W : Type u
    inst✝¹ : AddCommGroup W
    inst✝ : Module (MonoidAlgebra k G) W
    f : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
    hf : Eq (LinearMap.ker f) Bot.bot
    A : Type u := MonoidAlgebra k G
    this✝ : Module k W := Module.compHom W (algebraMap k A)
    this : Module k V := Module.compHom V (algebraMap k A)
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  have := IsScalarTower.of_compHom k A W
  /-
    k : Type u
    inst✝⁷ : Field k
    G : Type u
    inst✝⁶ : Fintype G
    inst✝⁵ : NeZero ↑(Fintype.card G)
    inst✝⁴ : Group G
    V : Type u
    inst✝³ : AddCommGroup V
    inst✝² : Module (MonoidAlgebra k G) V
    W : Type u
    inst✝¹ : AddCommGroup W
    inst✝ : Module (MonoidAlgebra k G) W
    f : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
    hf : Eq (LinearMap.ker f) Bot.bot
    A : Type u := MonoidAlgebra k G
    this✝¹ : Module k W := Module.compHom W (algebraMap k A)
    this✝ : Module k V := Module.compHom V (algebraMap k A)
    this : IsScalarTower k A W
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  have := IsScalarTower.of_compHom k A V
  obtain ⟨φ, hφ⟩ := (f.restrictScalars k).exists_leftInverse_of_injective <| by
    simp only [hf, Submodule.restrictScalars_bot, LinearMap.ker_restrictScalars]
  /-
    case intro
    k : Type u
    inst✝⁷ : Field k
    G : Type u
    inst✝⁶ : Fintype G
    inst✝⁵ : NeZero ↑(Fintype.card G)
    inst✝⁴ : Group G
    V : Type u
    inst✝³ : AddCommGroup V
    inst✝² : Module (MonoidAlgebra k G) V
    W : Type u
    inst✝¹ : AddCommGroup W
    inst✝ : Module (MonoidAlgebra k G) W
    f : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
    hf : Eq (LinearMap.ker f) Bot.bot
    A : Type u := MonoidAlgebra k G
    this✝² : Module k W := Module.compHom W (algebraMap k A)
    this✝¹ : Module k V := Module.compHom V (algebraMap k A)
    this✝ : IsScalarTower k A W
    this : IsScalarTower k A V
    φ : LinearMap (RingHom.id k) W V
    hφ : Eq (φ.comp (↑k f)) LinearMap.id
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  refine ⟨φ.equivariantProjection G, DFunLike.ext _ _ ?_⟩
  /-
    case intro
    k : Type u
    inst✝⁷ : Field k
    G : Type u
    inst✝⁶ : Fintype G
    inst✝⁵ : NeZero ↑(Fintype.card G)
    inst✝⁴ : Group G
    V : Type u
    inst✝³ : AddCommGroup V
    inst✝² : Module (MonoidAlgebra k G) V
    W : Type u
    inst✝¹ : AddCommGroup W
    inst✝ : Module (MonoidAlgebra k G) W
    f : LinearMap (RingHom.id (MonoidAlgebra k G)) V W
    hf : Eq (LinearMap.ker f) Bot.bot
    A : Type u := MonoidAlgebra k G
    this✝² : Module k W := Module.compHom W (algebraMap k A)
    this✝¹ : Module k V := Module.compHom V (algebraMap k A)
    this✝ : IsScalarTower k A W
    this : IsScalarTower k A V
    φ : LinearMap (RingHom.id k) W V
    hφ : Eq (φ.comp (↑k f)) LinearMap.id
    ⊢ ∀ (x : V), Eq (((LinearMap.equivariantProjection G φ).comp f) x) (LinearMap. …
  -/
  exact φ.equivariantProjection_condition G _ (.mk0 _ <| NeZero.ne _) <| DFunLike.congr_fun hφ
  /-
    🎉 no goals
  -/


theorem exists_isCompl (p : Submodule (MonoidAlgebra k G) V) :
    ∃ q : Submodule (MonoidAlgebra k G) V, IsCompl p q := by
  /-
    k : Type u
    inst✝⁵ : Field k
    G : Type u
    inst✝⁴ : Fintype G
    inst✝³ : NeZero ↑(Fintype.card G)
    inst✝² : Group G
    V : Type u
    inst✝¹ : AddCommGroup V
    inst✝ : Module (MonoidAlgebra k G) V
    p : Submodule (MonoidAlgebra k G) V
    ⊢ Exists fun q => IsCompl p q
  -/
  rcases MonoidAlgebra.exists_leftInverse_of_injective p.subtype p.ker_subtype with ⟨f, hf⟩
  /-
    case intro
    k : Type u
    inst✝⁵ : Field k
    G : Type u
    inst✝⁴ : Fintype G
    inst✝³ : NeZero ↑(Fintype.card G)
    inst✝² : Group G
    V : Type u
    inst✝¹ : AddCommGroup V
    inst✝ : Module (MonoidAlgebra k G) V
    p : Submodule (MonoidAlgebra k G) V
    f : LinearMap (RingHom.id (MonoidAlgebra k G)) V (Subtype fun x => Membership. …
    hf : Eq (f.comp p.subtype) LinearMap.id
    ⊢ Exists fun q => IsCompl p q
  -/
  exact ⟨LinearMap.ker f, LinearMap.isCompl_of_proj <| DFunLike.congr_fun hf⟩
  /-
    🎉 no goals
  -/


/-- This also implies instances `IsSemisimpleModule (MonoidAlgebra k G) V` and
`IsSemisimpleRing (MonoidAlgebra k G)`. -/
instance complementedLattice : ComplementedLattice (Submodule (MonoidAlgebra k G) V) :=
  ⟨exists_isCompl⟩


instance [AddGroup G] : IsSemisimpleRing (AddMonoidAlgebra k G) :=
  haveI : NeZero (Fintype.card (Multiplicative G) : k) := by
    /-
      k : Type u
      inst✝⁸ : Field k
      G : Type u
      inst✝⁷ : Fintype G
      inst✝⁶ : NeZero ↑(Fintype.card G)
      inst✝⁵ : Group G
      V : Type u
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module (MonoidAlgebra k G) V
      W : Type u
      inst✝² : AddCommGroup W
      inst✝¹ : Module (MonoidAlgebra k G) W
      inst✝ : AddGroup G
      ⊢ NeZero ↑(Fintype.card (Multiplicative G))
    -/
    rwa [Fintype.card_congr Multiplicative.toAdd]
    /-
      🎉 no goals
    -/
  (AddMonoidAlgebra.toMultiplicativeAlgEquiv k G (R := ℕ)).toRingEquiv.symm.isSemisimpleRing


