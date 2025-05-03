/-- The frobenius map of an algebra as a frobenius-semilinear map. -/
nonrec def LinearMap.frobenius [Algebra R S] : S →ₛₗ[frobenius R p] S where
  __ := frobenius S p
  map_smul' r s := show frobenius S p _ = _ by
    /-
      R : Type u
      inst✝⁴ : CommSemiring R
      S : Type u_1
      inst✝³ : CommSemiring S
      f : MonoidHom R S
      g : RingHom R S
      p m n : Nat
      inst✝² : ExpChar R p
      inst✝¹ : ExpChar S p
      x y : R
      inst✝ : Algebra R S
      r : R
      s : S
      ⊢ Eq ((_root_.frobenius S p) (HSMul.hSMul r s)) (HSMul.hSMul ((_root_.frobeniu …
    -/
    simp_rw [Algebra.smul_def, map_mul, ← (algebraMap R S).map_frobenius]; rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- The iterated frobenius map of an algebra as a iterated-frobenius-semilinear map. -/
nonrec def LinearMap.iterateFrobenius [Algebra R S] : S →ₛₗ[iterateFrobenius R p n] S where
  __ := iterateFrobenius S p n
  map_smul' f s := show iterateFrobenius S p n _ = _ by
    /-
      R : Type u
      inst✝⁴ : CommSemiring R
      S : Type u_1
      inst✝³ : CommSemiring S
      f✝ : MonoidHom R S
      g : RingHom R S
      p m n : Nat
      inst✝² : ExpChar R p
      inst✝¹ : ExpChar S p
      x y : R
      inst✝ : Algebra R S
      f : R
      s : S
      ⊢ Eq ((_root_.iterateFrobenius S p n) (HSMul.hSMul f s)) (HSMul.hSMul ((_root_ …
    -/
    simp_rw [iterateFrobenius_def, Algebra.smul_def, mul_pow, ← map_pow]; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem LinearMap.frobenius_def [Algebra R S] (x : S) : frobenius R S p x = x ^ p := rfl


theorem LinearMap.iterateFrobenius_def [Algebra R S] (n : ℕ) (x : S) :
    iterateFrobenius R S p n x = x ^ p ^ n := rfl


theorem frobenius_zero : frobenius R p 0 = 0 :=
  (frobenius R p).map_zero


theorem frobenius_add : frobenius R p (x + y) = frobenius R p x + frobenius R p y :=
  (frobenius R p).map_add x y


theorem frobenius_natCast (n : ℕ) : frobenius R p n = n :=
  map_natCast (frobenius R p) n


@[deprecated (since := "2024-04-17")]
alias frobenius_nat_cast := frobenius_natCast


