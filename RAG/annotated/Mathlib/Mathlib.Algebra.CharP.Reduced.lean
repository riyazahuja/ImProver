theorem iterateFrobenius_inj : Function.Injective (iterateFrobenius R p n) := fun x y H ↦ by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsReduced R
    p n : Nat
    inst✝ : ExpChar R p
    x y : R
    H : Eq ((iterateFrobenius R p n) x) ((iterateFrobenius R p n) y)
    ⊢ Eq x y
  -/
  rw [← sub_eq_zero] at H ⊢
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsReduced R
    p n : Nat
    inst✝ : ExpChar R p
    x y : R
    H : Eq (HSub.hSub ((iterateFrobenius R p n) x) ((iterateFrobenius R p n) y)) 0
    ⊢ Eq (HSub.hSub x y) 0
  -/
  simp_rw [iterateFrobenius_def, ← sub_pow_expChar_pow] at H
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsReduced R
    p n : Nat
    inst✝ : ExpChar R p
    x y : R
    H : Eq (HPow.hPow (HSub.hSub x y) (HPow.hPow p n)) 0
    ⊢ Eq (HSub.hSub x y) 0
  -/
  exact IsReduced.eq_zero _ ⟨_, H⟩
  /-
    🎉 no goals
  -/


theorem frobenius_inj : Function.Injective (frobenius R p) :=
  iterateFrobenius_one (R := R) p ▸ iterateFrobenius_inj R p 1


/-- If `ringChar R = 2`, where `R` is a finite reduced commutative ring,
then every `a : R` is a square. -/
theorem isSquare_of_charTwo' {R : Type*} [Finite R] [CommRing R] [IsReduced R] [CharP R 2]
    (a : R) : IsSquare a := by
  /-
    R : Type u_1
    inst✝³ : Finite R
    inst✝² : CommRing R
    inst✝¹ : IsReduced R
    inst✝ : CharP R 2
    a : R
    ⊢ IsSquare a
  -/
  cases nonempty_fintype R
  exact
    Exists.imp (fun b h => pow_two b ▸ Eq.symm h)
      (((Fintype.bijective_iff_injective_and_card _).mpr ⟨frobenius_inj R 2, rfl⟩).surjective a)


@[simp]
theorem ExpChar.pow_prime_pow_mul_eq_one_iff (p k m : ℕ) [ExpChar R p] (x : R) :
    x ^ (p ^ k * m) = 1 ↔ x ^ m = 1 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsReduced R
    p k m : Nat
    inst✝ : ExpChar R p
    x : R
    ⊢ Iff (Eq (HPow.hPow x (HMul.hMul (HPow.hPow p k) m)) 1) (Eq (HPow.hPow x m) 1)
  -/
  rw [pow_mul']
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsReduced R
    p k m : Nat
    inst✝ : ExpChar R p
    x : R
    ⊢ Iff (Eq (HPow.hPow (HPow.hPow x m) (HPow.hPow p k)) 1) (Eq (HPow.hPow x m) 1)
  -/
  convert ← (iterateFrobenius_inj R p k).eq_iff
  /-
    case h.e'_1.h.e'_3
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsReduced R
    p k m : Nat
    inst✝ : ExpChar R p
    x : R
    ⊢ Eq ((iterateFrobenius R p k) 1) 1
  -/
  apply map_one
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")]
alias CharP.pow_prime_pow_mul_eq_one_iff := ExpChar.pow_prime_pow_mul_eq_one_iff

