/-- A vertex operator over a commutative ring `R` is an `R`-linear map from an `R`-module `V` to
Laurent series with coefficients in `V`.  We write this as a specialization of the heterogeneous
case. -/
abbrev VertexOperator (R : Type*) (V : Type*) [CommRing R] [AddCommGroup V]
    [Module R V] := HVertexOperator ℤ R V V


@[ext]
theorem ext (A B : VertexOperator R V) (h : ∀ v : V, A v = B v) :
    A = B := LinearMap.ext h


/-- The coefficient of a vertex operator under normalized indexing. -/
def ncoeff {R} [CommRing R] [AddCommGroup V] [Module R V] (A : VertexOperator R V) (n : ℤ) :
    Module.End R V := HVertexOperator.coeff A (-n - 1)


/-- In the literature, the `n`th normalized coefficient of a vertex operator `A` is written as
either `Aₙ` or `A(n)`. -/
scoped[VertexOperator] notation A "[[" n "]]" => ncoeff A n


@[simp]
theorem coeff_eq_ncoeff (A : VertexOperator R V)
    (n : ℤ) : HVertexOperator.coeff A n = A [[-n - 1]] := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    A : VertexOperator R V
    n : Int
    ⊢ Eq (HVertexOperator.coeff A n) (A.ncoeff (HSub.hSub (Neg.neg n) 1))
  -/
  rw [ncoeff, neg_sub, sub_neg_eq_add, add_sub_cancel_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem ncoeff_add (A B : VertexOperator R V) (n : ℤ) : (A + B) [[n]] = A [[n]] + B [[n]] := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    A B : VertexOperator R V
    n : Int
    ⊢ Eq ((HAdd.hAdd A B).ncoeff n) (HAdd.hAdd (A.ncoeff n) (B.ncoeff n))
  -/
  rw [ncoeff, ncoeff, ncoeff, add_coeff, Pi.add_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem ncoeff_smul (A : VertexOperator R V) (r : R) (n : ℤ) : (r • A) [[n]] = r • A [[n]] := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    A : VertexOperator R V
    r : R
    n : Int
    ⊢ Eq ((HSMul.hSMul r A).ncoeff n) (HSMul.hSMul r (A.ncoeff n))
  -/
  rw [ncoeff, ncoeff, smul_coeff, Pi.smul_apply]
  /-
    🎉 no goals
  -/


theorem ncoeff_eq_zero_of_lt_order (A : VertexOperator R V) (n : ℤ) (x : V)
    (h : -n - 1 < HahnSeries.order ((HahnModule.of R).symm (A x))) : (A [[n]]) x = 0 := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    A : VertexOperator R V
    n : Int
    x : V
    h : LT.lt (HSub.hSub (Neg.neg n) 1) ((HahnModule.of R).symm (A x)).order
    ⊢ Eq ((A.ncoeff n) x) 0
  -/
  simp only [ncoeff, HVertexOperator.coeff, LinearMap.coe_mk, AddHom.coe_mk]
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    A : VertexOperator R V
    n : Int
    x : V
    h : LT.lt (HSub.hSub (Neg.neg n) 1) ((HahnModule.of R).symm (A x)).order
    ⊢ Eq (((HahnModule.of R).symm (A x)).coeff (HSub.hSub (Neg.neg n) 1)) 0
  -/
  exact HahnSeries.coeff_eq_zero_of_lt_order h
  /-
    🎉 no goals
  -/


theorem coeff_eq_zero_of_lt_order (A : VertexOperator R V) (n : ℤ) (x : V)
    (h : n < HahnSeries.order ((HahnModule.of R).symm (A x))) : coeff A n x = 0 := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    A : VertexOperator R V
    n : Int
    x : V
    h : LT.lt n ((HahnModule.of R).symm (A x)).order
    ⊢ Eq ((HVertexOperator.coeff A n) x) 0
  -/
  rw [coeff_eq_ncoeff, ncoeff_eq_zero_of_lt_order A (-n - 1) x]
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    A : VertexOperator R V
    n : Int
    x : V
    h : LT.lt n ((HahnModule.of R).symm (A x)).order
    ⊢ LT.lt (HSub.hSub (Neg.neg (HSub.hSub (Neg.neg n) 1)) 1) ((HahnModule.of R).s …
  -/
  omega
  /-
    🎉 no goals
  -/


/-- Given an endomorphism-valued function on integers satisfying a pointwise bounded-pole condition,
we produce a vertex operator. -/
noncomputable def of_coeff (f : ℤ → Module.End R V)
    (hf : ∀ (x : V), ∃ (n : ℤ), ∀ (m : ℤ), m < n → (f m) x = 0) : VertexOperator R V :=
  HVertexOperator.of_coeff f
    (fun x => HahnSeries.suppBddBelow_supp_PWO (fun n => (f n) x)
      (HahnSeries.forallLTEqZero_supp_BddBelow (fun n => (f n) x)
        (Exists.choose (hf x)) (Exists.choose_spec (hf x))))


@[simp]
theorem of_coeff_apply_coeff (f : ℤ → Module.End R V)
    (hf : ∀ (x : V), ∃ n, ∀ m < n, (f m) x = 0) (x : V) (n : ℤ) :
    ((HahnModule.of R).symm ((of_coeff f hf) x)).coeff n = (f n) x := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    f : Int → Module.End R V
    hf : ∀ (x : V), Exists fun n => ∀ (m : Int), LT.lt m n → Eq ((f m) x) 0
    x : V
    n : Int
    ⊢ Eq (((HahnModule.of R).symm ((VertexOperator.of_coeff f hf) x)).coeff n) ((f …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ncoeff_of_coeff (f : ℤ → Module.End R V)
    (hf : ∀(x : V), ∃(n : ℤ), ∀(m : ℤ), m < n → (f m) x = 0) (n : ℤ) :
    (of_coeff f hf) [[n]] = f (-n - 1) := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    f : Int → Module.End R V
    hf : ∀ (x : V), Exists fun n => ∀ (m : Int), LT.lt m n → Eq ((f m) x) 0
    n : Int
    ⊢ Eq ((VertexOperator.of_coeff f hf).ncoeff n) (f (HSub.hSub (Neg.neg n) 1))
  -/
  ext v
  /-
    case h
    R : Type u_1
    V : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    f : Int → Module.End R V
    hf : ∀ (x : V), Exists fun n => ∀ (m : Int), LT.lt m n → Eq ((f m) x) 0
    n : Int
    v : V
    ⊢ Eq (((VertexOperator.of_coeff f hf).ncoeff n) v) ((f (HSub.hSub (Neg.neg n)  …
  -/
  rw [ncoeff, coeff_apply, of_coeff_apply_coeff]
  /-
    🎉 no goals
  -/


