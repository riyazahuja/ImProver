/-- A heterogeneous `Γ`-vertex operator over a commutator ring `R` is an `R`-linear map from an
`R`-module `V` to `Γ`-Hahn series with coefficients in an `R`-module `W`. -/
abbrev HVertexOperator (Γ : Type*) [PartialOrder Γ] (R : Type*) [CommRing R]
    (V : Type*) (W : Type*) [AddCommGroup V] [Module R V] [AddCommGroup W] [Module R W] :=
  V →ₗ[R] (HahnModule Γ R W)


@[ext]
theorem ext (A B : HVertexOperator Γ R V W) (h : ∀ v : V, A v = B v) :
    A = B := LinearMap.ext h


@[deprecated (since := "2024-06-18")] alias _root_.VertexAlg.HetVertexOperator.ext := ext


/-- The coefficient of a heterogeneous vertex operator, viewed as a formal power series with
coefficients in linear maps. -/
@[simps]
def coeff (A : HVertexOperator Γ R V W) (n : Γ) : V →ₗ[R] W where
  toFun v := ((of R).symm (A v)).coeff n
                     /-
                       Γ : Type u_1
                       inst✝⁵ : PartialOrder Γ
                       R : Type u_2
                       V : Type u_3
                       W : Type u_4
                       inst✝⁴ : CommRing R
                       inst✝³ : AddCommGroup V
                       inst✝² : Module R V
                       inst✝¹ : AddCommGroup W
                       inst✝ : Module R W
                       A : HVertexOperator Γ R V W
                       n : Γ
                       x✝¹ x✝ : V
                       ⊢ Eq ((fun v => ((HahnModule.of R).symm (A v)).coeff n) (HAdd.hAdd x✝¹ x✝)) (H …
                     -/
  map_add' _ _ := by simp
                     /-
                       🎉 no goals
                     -/
  map_smul' _ _ := by
    /-
      Γ : Type u_1
      inst✝⁵ : PartialOrder Γ
      R : Type u_2
      V : Type u_3
      W : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddCommGroup W
      inst✝ : Module R W
      A : HVertexOperator Γ R V W
      n : Γ
      x✝¹ : R
      x✝ : V
      ⊢ Eq ({ toFun := fun v => ((HahnModule.of R).symm (A v)).coeff n, map_add' :=  …
    -/
    simp only [map_smul, RingHom.id_apply, of_symm_smul, HahnSeries.smul_coeff]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-18")] alias _root_.VertexAlg.coeff := coeff


theorem coeff_isPWOsupport (A : HVertexOperator Γ R V W) (v : V) :
    ((of R).symm (A v)).coeff.support.IsPWO :=
  ((of R).symm (A v)).isPWO_support'


@[deprecated (since := "2024-06-18")]
alias _root_.VertexAlg.coeff_isPWOsupport := coeff_isPWOsupport


@[ext]
theorem coeff_inj : Function.Injective (coeff : HVertexOperator Γ R V W → Γ → (V →ₗ[R] W)) := by
  /-
    Γ : Type u_1
    inst✝⁵ : PartialOrder Γ
    R : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    ⊢ Function.Injective HVertexOperator.coeff
  -/
  intro _ _ h
  /-
    Γ : Type u_1
    inst✝⁵ : PartialOrder Γ
    R : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    a₁✝ a₂✝ : HVertexOperator Γ R V W
    h : Eq a₁✝.coeff a₂✝.coeff
    ⊢ Eq a₁✝ a₂✝
  -/
  ext v n
  /-
    case h.h.h
    Γ : Type u_1
    inst✝⁵ : PartialOrder Γ
    R : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    a₁✝ a₂✝ : HVertexOperator Γ R V W
    h : Eq a₁✝.coeff a₂✝.coeff
    v : V
    n : Γ
    ⊢ Eq (((HahnModule.of R).symm (a₁✝ v)).coeff n) (((HahnModule.of R).symm (a₂✝  …
  -/
  exact congrFun (congrArg DFunLike.coe (congrFun h n)) v
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-18")] alias _root_.VertexAlg.coeff_inj := coeff_inj


/-- Given a coefficient function valued in linear maps satisfying a partially well-ordered support
condition, we produce a heterogeneous vertex operator. -/
@[simps]
def of_coeff (f : Γ → V →ₗ[R] W)
    (hf : ∀(x : V), (Function.support (f · x)).IsPWO) : HVertexOperator Γ R V W where
  toFun x := (of R) { coeff := fun g => f g x, isPWO_support' := hf x }
                     /-
                       Γ : Type u_1
                       inst✝⁵ : PartialOrder Γ
                       R : Type u_2
                       V : Type u_3
                       W : Type u_4
                       inst✝⁴ : CommRing R
                       inst✝³ : AddCommGroup V
                       inst✝² : Module R V
                       inst✝¹ : AddCommGroup W
                       inst✝ : Module R W
                       f : Γ → LinearMap (RingHom.id R) V W
                       hf : ∀ (x : V), (Function.support fun x_1 => (f x_1) x).IsPWO
                       x✝¹ x✝ : V
                       ⊢ Eq ((fun x => (HahnModule.of R) { coeff := fun g => (f g) x, isPWO_support'  …
                     -/
  map_add' _ _ := by ext; simp
                          /-
                            🎉 no goals
                          -/
                      /-
                        Γ : Type u_1
                        inst✝⁵ : PartialOrder Γ
                        R : Type u_2
                        V : Type u_3
                        W : Type u_4
                        inst✝⁴ : CommRing R
                        inst✝³ : AddCommGroup V
                        inst✝² : Module R V
                        inst✝¹ : AddCommGroup W
                        inst✝ : Module R W
                        f : Γ → LinearMap (RingHom.id R) V W
                        hf : ∀ (x : V), (Function.support fun x_1 => (f x_1) x).IsPWO
                        x✝¹ : R
                        x✝ : V
                        ⊢ Eq ({ toFun := fun x => (HahnModule.of R) { coeff := fun g => (f g) x, isPWO …
                      -/
  map_smul' _ _ := by ext; simp
                           /-
                             🎉 no goals
                           -/


@[deprecated (since := "2024-06-18")] alias _root_.VertexAlg.HetVertexOperator.of_coeff := of_coeff


@[simp]
theorem add_coeff (A B : HVertexOperator Γ R V W) : (A + B).coeff = A.coeff + B.coeff := by
  /-
    Γ : Type u_1
    inst✝⁵ : PartialOrder Γ
    R : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    A B : HVertexOperator Γ R V W
    ⊢ Eq (HAdd.hAdd A B).coeff (HAdd.hAdd A.coeff B.coeff)
  -/
  ext
  /-
    case h.h
    Γ : Type u_1
    inst✝⁵ : PartialOrder Γ
    R : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    A B : HVertexOperator Γ R V W
    x✝¹ : Γ
    x✝ : V
    ⊢ Eq (((HAdd.hAdd A B).coeff x✝¹) x✝) ((HAdd.hAdd A.coeff B.coeff x✝¹) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_coeff (A : HVertexOperator Γ R V W) (r : R) : (r • A).coeff = r • (A.coeff) := by
  /-
    Γ : Type u_1
    inst✝⁵ : PartialOrder Γ
    R : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    A : HVertexOperator Γ R V W
    r : R
    ⊢ Eq (HSMul.hSMul r A).coeff (HSMul.hSMul r A.coeff)
  -/
  ext
  /-
    case h.h
    Γ : Type u_1
    inst✝⁵ : PartialOrder Γ
    R : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    A : HVertexOperator Γ R V W
    r : R
    x✝¹ : Γ
    x✝ : V
    ⊢ Eq (((HSMul.hSMul r A).coeff x✝¹) x✝) ((HSMul.hSMul r A.coeff x✝¹) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The composite of two heterogeneous vertex operators acting on a vector, as an iterated Hahn
series. -/
@[simps]
def compHahnSeries (u : U) : HahnSeries Γ' (HahnSeries Γ W) where
  coeff g' := A (coeff B g' u)
  isPWO_support' := by
    /-
      Γ✝ : Type u_1
      inst✝¹⁴ : PartialOrder Γ✝
      R✝ : Type u_2
      V✝ : Type u_3
      W✝ : Type u_4
      inst✝¹³ : CommRing R✝
      inst✝¹² : AddCommGroup V✝
      inst✝¹¹ : Module R✝ V✝
      inst✝¹⁰ : AddCommGroup W✝
      inst✝⁹ : Module R✝ W✝
      Γ : Type u_5
      Γ' : Type u_6
      inst✝⁸ : OrderedCancelAddCommMonoid Γ
      inst✝⁷ : OrderedCancelAddCommMonoid Γ'
      R : Type u_7
      inst✝⁶ : CommRing R
      U : Type u_8
      V : Type u_9
      W : Type u_10
      inst✝⁵ : AddCommGroup U
      inst✝⁴ : Module R U
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddCommGroup W
      inst✝ : Module R W
      A : HVertexOperator Γ R V W
      B : HVertexOperator Γ' R U V
      u : U
      ⊢ (Function.support fun g' => A ((B.coeff g') u)).IsPWO
    -/
    refine Set.IsPWO.mono (((of R).symm (B u)).isPWO_support') ?_
    /-
      Γ✝ : Type u_1
      inst✝¹⁴ : PartialOrder Γ✝
      R✝ : Type u_2
      V✝ : Type u_3
      W✝ : Type u_4
      inst✝¹³ : CommRing R✝
      inst✝¹² : AddCommGroup V✝
      inst✝¹¹ : Module R✝ V✝
      inst✝¹⁰ : AddCommGroup W✝
      inst✝⁹ : Module R✝ W✝
      Γ : Type u_5
      Γ' : Type u_6
      inst✝⁸ : OrderedCancelAddCommMonoid Γ
      inst✝⁷ : OrderedCancelAddCommMonoid Γ'
      R : Type u_7
      inst✝⁶ : CommRing R
      U : Type u_8
      V : Type u_9
      W : Type u_10
      inst✝⁵ : AddCommGroup U
      inst✝⁴ : Module R U
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddCommGroup W
      inst✝ : Module R W
      A : HVertexOperator Γ R V W
      B : HVertexOperator Γ' R U V
      u : U
      ⊢ HasSubset.Subset (Function.support fun g' => A ((B.coeff g') u)) (Function.s …
    -/
    simp_all only [coeff_apply, Function.support_subset_iff, ne_eq, Function.mem_support]
    /-
      Γ✝ : Type u_1
      inst✝¹⁴ : PartialOrder Γ✝
      R✝ : Type u_2
      V✝ : Type u_3
      W✝ : Type u_4
      inst✝¹³ : CommRing R✝
      inst✝¹² : AddCommGroup V✝
      inst✝¹¹ : Module R✝ V✝
      inst✝¹⁰ : AddCommGroup W✝
      inst✝⁹ : Module R✝ W✝
      Γ : Type u_5
      Γ' : Type u_6
      inst✝⁸ : OrderedCancelAddCommMonoid Γ
      inst✝⁷ : OrderedCancelAddCommMonoid Γ'
      R : Type u_7
      inst✝⁶ : CommRing R
      U : Type u_8
      V : Type u_9
      W : Type u_10
      inst✝⁵ : AddCommGroup U
      inst✝⁴ : Module R U
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddCommGroup W
      inst✝ : Module R W
      A : HVertexOperator Γ R V W
      B : HVertexOperator Γ' R U V
      u : U
      ⊢ ∀ (x : Γ'), Not (Eq (A (((HahnModule.of R).symm (B u)).coeff x)) 0) → Not (E …
    -/
    intro g' hg' hAB
    /-
      Γ✝ : Type u_1
      inst✝¹⁴ : PartialOrder Γ✝
      R✝ : Type u_2
      V✝ : Type u_3
      W✝ : Type u_4
      inst✝¹³ : CommRing R✝
      inst✝¹² : AddCommGroup V✝
      inst✝¹¹ : Module R✝ V✝
      inst✝¹⁰ : AddCommGroup W✝
      inst✝⁹ : Module R✝ W✝
      Γ : Type u_5
      Γ' : Type u_6
      inst✝⁸ : OrderedCancelAddCommMonoid Γ
      inst✝⁷ : OrderedCancelAddCommMonoid Γ'
      R : Type u_7
      inst✝⁶ : CommRing R
      U : Type u_8
      V : Type u_9
      W : Type u_10
      inst✝⁵ : AddCommGroup U
      inst✝⁴ : Module R U
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddCommGroup W
      inst✝ : Module R W
      A : HVertexOperator Γ R V W
      B : HVertexOperator Γ' R U V
      u : U
      g' : Γ'
      hg' : Not (Eq (A (((HahnModule.of R).symm (B u)).coeff g')) 0)
      hAB : Eq (((HahnModule.of R).symm (B u)).coeff g') 0
      ⊢ False
    -/
    apply hg'
    /-
      Γ✝ : Type u_1
      inst✝¹⁴ : PartialOrder Γ✝
      R✝ : Type u_2
      V✝ : Type u_3
      W✝ : Type u_4
      inst✝¹³ : CommRing R✝
      inst✝¹² : AddCommGroup V✝
      inst✝¹¹ : Module R✝ V✝
      inst✝¹⁰ : AddCommGroup W✝
      inst✝⁹ : Module R✝ W✝
      Γ : Type u_5
      Γ' : Type u_6
      inst✝⁸ : OrderedCancelAddCommMonoid Γ
      inst✝⁷ : OrderedCancelAddCommMonoid Γ'
      R : Type u_7
      inst✝⁶ : CommRing R
      U : Type u_8
      V : Type u_9
      W : Type u_10
      inst✝⁵ : AddCommGroup U
      inst✝⁴ : Module R U
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddCommGroup W
      inst✝ : Module R W
      A : HVertexOperator Γ R V W
      B : HVertexOperator Γ' R U V
      u : U
      g' : Γ'
      hg' : Not (Eq (A (((HahnModule.of R).symm (B u)).coeff g')) 0)
      hAB : Eq (((HahnModule.of R).symm (B u)).coeff g') 0
      ⊢ Eq (A (((HahnModule.of R).symm (B u)).coeff g')) 0
    -/
    simp_rw [hAB]
    /-
      Γ✝ : Type u_1
      inst✝¹⁴ : PartialOrder Γ✝
      R✝ : Type u_2
      V✝ : Type u_3
      W✝ : Type u_4
      inst✝¹³ : CommRing R✝
      inst✝¹² : AddCommGroup V✝
      inst✝¹¹ : Module R✝ V✝
      inst✝¹⁰ : AddCommGroup W✝
      inst✝⁹ : Module R✝ W✝
      Γ : Type u_5
      Γ' : Type u_6
      inst✝⁸ : OrderedCancelAddCommMonoid Γ
      inst✝⁷ : OrderedCancelAddCommMonoid Γ'
      R : Type u_7
      inst✝⁶ : CommRing R
      U : Type u_8
      V : Type u_9
      W : Type u_10
      inst✝⁵ : AddCommGroup U
      inst✝⁴ : Module R U
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddCommGroup W
      inst✝ : Module R W
      A : HVertexOperator Γ R V W
      B : HVertexOperator Γ' R U V
      u : U
      g' : Γ'
      hg' : Not (Eq (A (((HahnModule.of R).symm (B u)).coeff g')) 0)
      hAB : Eq (((HahnModule.of R).symm (B u)).coeff g') 0
      ⊢ Eq (A 0) 0
    -/
    simp_all only [map_zero, HahnSeries.zero_coeff, not_true_eq_false]
    /-
      🎉 no goals
    -/


@[simp]
theorem compHahnSeries_add (u v : U) :
    compHahnSeries A B (u + v) = compHahnSeries A B u + compHahnSeries A B v := by
  /-
    Γ : Type u_5
    Γ' : Type u_6
    inst✝⁸ : OrderedCancelAddCommMonoid Γ
    inst✝⁷ : OrderedCancelAddCommMonoid Γ'
    R : Type u_7
    inst✝⁶ : CommRing R
    U : Type u_8
    V : Type u_9
    W : Type u_10
    inst✝⁵ : AddCommGroup U
    inst✝⁴ : Module R U
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    A : HVertexOperator Γ R V W
    B : HVertexOperator Γ' R U V
    u v : U
    ⊢ Eq (A.compHahnSeries B (HAdd.hAdd u v)) (HAdd.hAdd (A.compHahnSeries B u) (A …
  -/
  ext
  /-
    case coeff.h.coeff.h
    Γ : Type u_5
    Γ' : Type u_6
    inst✝⁸ : OrderedCancelAddCommMonoid Γ
    inst✝⁷ : OrderedCancelAddCommMonoid Γ'
    R : Type u_7
    inst✝⁶ : CommRing R
    U : Type u_8
    V : Type u_9
    W : Type u_10
    inst✝⁵ : AddCommGroup U
    inst✝⁴ : Module R U
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    A : HVertexOperator Γ R V W
    B : HVertexOperator Γ' R U V
    u v : U
    x✝¹ : Γ'
    x✝ : Γ
    ⊢ Eq (((A.compHahnSeries B (HAdd.hAdd u v)).coeff x✝¹).coeff x✝) (((HAdd.hAdd  …
  -/
  simp only [compHahnSeries_coeff, map_add, coeff_apply, HahnSeries.add_coeff', Pi.add_apply]
  /-
    case coeff.h.coeff.h
    Γ : Type u_5
    Γ' : Type u_6
    inst✝⁸ : OrderedCancelAddCommMonoid Γ
    inst✝⁷ : OrderedCancelAddCommMonoid Γ'
    R : Type u_7
    inst✝⁶ : CommRing R
    U : Type u_8
    V : Type u_9
    W : Type u_10
    inst✝⁵ : AddCommGroup U
    inst✝⁴ : Module R U
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    A : HVertexOperator Γ R V W
    B : HVertexOperator Γ' R U V
    u v : U
    x✝¹ : Γ'
    x✝ : Γ
    ⊢ Eq ((HAdd.hAdd (A (((HahnModule.of R).symm (B u)).coeff x✝¹)) (A (((HahnModu …
  -/
  rw [← HahnSeries.add_coeff]
  /-
    🎉 no goals
  -/


@[simp]
theorem compHahnSeries_smul (r : R) (u : U) :
    compHahnSeries A B (r • u) = r • compHahnSeries A B u := by
  /-
    Γ : Type u_5
    Γ' : Type u_6
    inst✝⁸ : OrderedCancelAddCommMonoid Γ
    inst✝⁷ : OrderedCancelAddCommMonoid Γ'
    R : Type u_7
    inst✝⁶ : CommRing R
    U : Type u_8
    V : Type u_9
    W : Type u_10
    inst✝⁵ : AddCommGroup U
    inst✝⁴ : Module R U
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    A : HVertexOperator Γ R V W
    B : HVertexOperator Γ' R U V
    r : R
    u : U
    ⊢ Eq (A.compHahnSeries B (HSMul.hSMul r u)) (HSMul.hSMul r (A.compHahnSeries B …
  -/
  ext
  /-
    case coeff.h.coeff.h
    Γ : Type u_5
    Γ' : Type u_6
    inst✝⁸ : OrderedCancelAddCommMonoid Γ
    inst✝⁷ : OrderedCancelAddCommMonoid Γ'
    R : Type u_7
    inst✝⁶ : CommRing R
    U : Type u_8
    V : Type u_9
    W : Type u_10
    inst✝⁵ : AddCommGroup U
    inst✝⁴ : Module R U
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    A : HVertexOperator Γ R V W
    B : HVertexOperator Γ' R U V
    r : R
    u : U
    x✝¹ : Γ'
    x✝ : Γ
    ⊢ Eq (((A.compHahnSeries B (HSMul.hSMul r u)).coeff x✝¹).coeff x✝) (((HSMul.hS …
  -/
  simp only [compHahnSeries_coeff, LinearMapClass.map_smul, coeff_apply, HahnSeries.smul_coeff]
  /-
    case coeff.h.coeff.h
    Γ : Type u_5
    Γ' : Type u_6
    inst✝⁸ : OrderedCancelAddCommMonoid Γ
    inst✝⁷ : OrderedCancelAddCommMonoid Γ'
    R : Type u_7
    inst✝⁶ : CommRing R
    U : Type u_8
    V : Type u_9
    W : Type u_10
    inst✝⁵ : AddCommGroup U
    inst✝⁴ : Module R U
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    A : HVertexOperator Γ R V W
    B : HVertexOperator Γ' R U V
    r : R
    u : U
    x✝¹ : Γ'
    x✝ : Γ
    ⊢ Eq ((HSMul.hSMul r (A (((HahnModule.of R).symm (B u)).coeff x✝¹))).coeff x✝) …
  -/
  rw [← HahnSeries.smul_coeff]
  /-
    🎉 no goals
  -/


/-- The composite of two heterogeneous vertex operators, as a heterogeneous vertex operator. -/
@[simps]
def comp : HVertexOperator (Γ' ×ₗ Γ) R U W where
  toFun u := HahnModule.of R (HahnSeries.ofIterate (compHahnSeries A B u))
  map_add' := by
    /-
      Γ✝ : Type u_1
      inst✝¹⁴ : PartialOrder Γ✝
      R✝ : Type u_2
      V✝ : Type u_3
      W✝ : Type u_4
      inst✝¹³ : CommRing R✝
      inst✝¹² : AddCommGroup V✝
      inst✝¹¹ : Module R✝ V✝
      inst✝¹⁰ : AddCommGroup W✝
      inst✝⁹ : Module R✝ W✝
      Γ : Type u_5
      Γ' : Type u_6
      inst✝⁸ : OrderedCancelAddCommMonoid Γ
      inst✝⁷ : OrderedCancelAddCommMonoid Γ'
      R : Type u_7
      inst✝⁶ : CommRing R
      U : Type u_8
      V : Type u_9
      W : Type u_10
      inst✝⁵ : AddCommGroup U
      inst✝⁴ : Module R U
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddCommGroup W
      inst✝ : Module R W
      A : HVertexOperator Γ R V W
      B : HVertexOperator Γ' R U V
      ⊢ ∀ (x y : U), Eq ((fun u => (HahnModule.of R) (A.compHahnSeries B u).ofIterat …
    -/
    intro u v
    /-
      Γ✝ : Type u_1
      inst✝¹⁴ : PartialOrder Γ✝
      R✝ : Type u_2
      V✝ : Type u_3
      W✝ : Type u_4
      inst✝¹³ : CommRing R✝
      inst✝¹² : AddCommGroup V✝
      inst✝¹¹ : Module R✝ V✝
      inst✝¹⁰ : AddCommGroup W✝
      inst✝⁹ : Module R✝ W✝
      Γ : Type u_5
      Γ' : Type u_6
      inst✝⁸ : OrderedCancelAddCommMonoid Γ
      inst✝⁷ : OrderedCancelAddCommMonoid Γ'
      R : Type u_7
      inst✝⁶ : CommRing R
      U : Type u_8
      V : Type u_9
      W : Type u_10
      inst✝⁵ : AddCommGroup U
      inst✝⁴ : Module R U
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddCommGroup W
      inst✝ : Module R W
      A : HVertexOperator Γ R V W
      B : HVertexOperator Γ' R U V
      u v : U
      ⊢ Eq ((fun u => (HahnModule.of R) (A.compHahnSeries B u).ofIterate) (HAdd.hAdd …
    -/
    ext g
    simp only [HahnSeries.ofIterate, compHahnSeries_add, Equiv.symm_apply_apply,
      HahnModule.of_symm_add, HahnSeries.add_coeff', Pi.add_apply]
  map_smul' := by
    /-
      Γ✝ : Type u_1
      inst✝¹⁴ : PartialOrder Γ✝
      R✝ : Type u_2
      V✝ : Type u_3
      W✝ : Type u_4
      inst✝¹³ : CommRing R✝
      inst✝¹² : AddCommGroup V✝
      inst✝¹¹ : Module R✝ V✝
      inst✝¹⁰ : AddCommGroup W✝
      inst✝⁹ : Module R✝ W✝
      Γ : Type u_5
      Γ' : Type u_6
      inst✝⁸ : OrderedCancelAddCommMonoid Γ
      inst✝⁷ : OrderedCancelAddCommMonoid Γ'
      R : Type u_7
      inst✝⁶ : CommRing R
      U : Type u_8
      V : Type u_9
      W : Type u_10
      inst✝⁵ : AddCommGroup U
      inst✝⁴ : Module R U
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddCommGroup W
      inst✝ : Module R W
      A : HVertexOperator Γ R V W
      B : HVertexOperator Γ' R U V
      ⊢ ∀ (m : R) (x : U), Eq ({ toFun := fun u => (HahnModule.of R) (A.compHahnSeri …
    -/
    intro r x
    /-
      Γ✝ : Type u_1
      inst✝¹⁴ : PartialOrder Γ✝
      R✝ : Type u_2
      V✝ : Type u_3
      W✝ : Type u_4
      inst✝¹³ : CommRing R✝
      inst✝¹² : AddCommGroup V✝
      inst✝¹¹ : Module R✝ V✝
      inst✝¹⁰ : AddCommGroup W✝
      inst✝⁹ : Module R✝ W✝
      Γ : Type u_5
      Γ' : Type u_6
      inst✝⁸ : OrderedCancelAddCommMonoid Γ
      inst✝⁷ : OrderedCancelAddCommMonoid Γ'
      R : Type u_7
      inst✝⁶ : CommRing R
      U : Type u_8
      V : Type u_9
      W : Type u_10
      inst✝⁵ : AddCommGroup U
      inst✝⁴ : Module R U
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddCommGroup W
      inst✝ : Module R W
      A : HVertexOperator Γ R V W
      B : HVertexOperator Γ' R U V
      r : R
      x : U
      ⊢ Eq ({ toFun := fun u => (HahnModule.of R) (A.compHahnSeries B u).ofIterate,  …
    -/
    ext g
    simp only [HahnSeries.ofIterate, compHahnSeries_smul, HahnSeries.smul_coeff,
      compHahnSeries_coeff, coeff_apply, Equiv.symm_apply_apply, RingHom.id_apply, of_symm_smul]


@[simp]
theorem comp_coeff (g : Γ' ×ₗ Γ) :
    (comp A B).coeff g = A.coeff (ofLex g).2 ∘ₗ B.coeff (ofLex g).1 := by
  /-
    Γ : Type u_5
    Γ' : Type u_6
    inst✝⁸ : OrderedCancelAddCommMonoid Γ
    inst✝⁷ : OrderedCancelAddCommMonoid Γ'
    R : Type u_7
    inst✝⁶ : CommRing R
    U : Type u_8
    V : Type u_9
    W : Type u_10
    inst✝⁵ : AddCommGroup U
    inst✝⁴ : Module R U
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddCommGroup W
    inst✝ : Module R W
    A : HVertexOperator Γ R V W
    B : HVertexOperator Γ' R U V
    g : Lex (Prod Γ' Γ)
    ⊢ Eq ((A.comp B).coeff g) ((A.coeff (ofLex g).2).comp (B.coeff (ofLex g).1))
  -/
  rfl
  /-
    🎉 no goals
  -/


