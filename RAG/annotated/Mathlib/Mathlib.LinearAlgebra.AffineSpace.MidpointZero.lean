theorem lineMap_inv_two {R : Type*} {V P : Type*} [DivisionRing R] [CharZero R] [AddCommGroup V]
    [Module R V] [AddTorsor V P] (a b : P) : lineMap a b (2⁻¹ : R) = midpoint R a b :=
  rfl


theorem lineMap_one_half {R : Type*} {V P : Type*} [DivisionRing R] [CharZero R] [AddCommGroup V]
    [Module R V] [AddTorsor V P] (a b : P) : lineMap a b (1 / 2 : R) = midpoint R a b := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing R
    inst✝³ : CharZero R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    a b : P
    ⊢ Eq ((AffineMap.lineMap a b) (1 / 2)) (midpoint R a b)
  -/
  rw [one_div, lineMap_inv_two]
  /-
    🎉 no goals
  -/


theorem homothety_invOf_two {R : Type*} {V P : Type*} [CommRing R] [Invertible (2 : R)]
    [AddCommGroup V] [Module R V] [AddTorsor V P] (a b : P) :
    homothety a (⅟ 2 : R) b = midpoint R a b :=
  rfl


theorem homothety_inv_two {k : Type*} {V P : Type*} [Field k] [CharZero k] [AddCommGroup V]
    [Module k V] [AddTorsor V P] (a b : P) : homothety a (2⁻¹ : k) b = midpoint k a b :=
  rfl


theorem homothety_one_half {k : Type*} {V P : Type*} [Field k] [CharZero k] [AddCommGroup V]
    [Module k V] [AddTorsor V P] (a b : P) : homothety a (1 / 2 : k) b = midpoint k a b := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : Field k
    inst✝³ : CharZero k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    a b : P
    ⊢ Eq ((AffineMap.homothety a (1 / 2)) b) (midpoint k a b)
  -/
  rw [one_div, homothety_inv_two]
  /-
    🎉 no goals
  -/


@[simp]
theorem pi_midpoint_apply {k ι : Type*} {V : ι → Type*} {P : ι → Type*} [Field k]
    [Invertible (2 : k)] [∀ i, AddCommGroup (V i)] [∀ i, Module k (V i)]
    [∀ i, AddTorsor (V i) (P i)] (f g : ∀ i, P i) (i : ι) :
    midpoint k f g i = midpoint k (f i) (g i) :=
  rfl

