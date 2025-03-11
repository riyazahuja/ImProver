local notation "𝕎" => WittVector p -- type as `\bbW`


/-- The underlying function of the monoid hom `WittVector.teichmuller`.
The `0`-th coefficient of `teichmullerFun p r` is `r`, and all others are `0`.
-/
def teichmullerFun (r : R) : 𝕎 R :=
  ⟨fun n => if n = 0 then r else 0⟩


private theorem ghostComponent_teichmullerFun (r : R) (n : ℕ) :
    ghostComponent n (teichmullerFun p r) = r ^ p ^ n := by
  rw [ghostComponent_apply, aeval_wittPolynomial, Finset.sum_eq_single 0, pow_zero, one_mul,
    tsub_zero]
    /-
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      r : R
      n : Nat
      ⊢ Eq (HPow.hPow ((WittVector.teichmullerFun p r).coeff 0) (HPow.hPow p n)) (HP …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h₀
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      r : R
      n : Nat
      ⊢ ∀ (b : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) b → Ne b 0 → Eq ( …
    -/
  · intro i _ h0
    /-
      case h₀
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      r : R
      n i : Nat
      a✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
      h0 : Ne i 0
      ⊢ Eq (HMul.hMul (HPow.hPow (↑p) i) (HPow.hPow ((WittVector.teichmullerFun p r) …
    -/
    simp [teichmullerFun, h0, hp.1.ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      r : R
      n : Nat
      ⊢ Not (Membership.mem (Finset.range (HAdd.hAdd n 1)) 0) → Eq (HMul.hMul (HPow. …
    -/
  · rw [Finset.mem_range]; intro h; exact (h (Nat.succ_pos n)).elim
                                    /-
                                      🎉 no goals
                                    -/


private theorem map_teichmullerFun (f : R →+* S) (r : R) :
    map f (teichmullerFun p r) = teichmullerFun p (f r) := by
  /-
    p : Nat
    R : Type u_1
    S : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    r : R
    ⊢ Eq ((WittVector.map f) (WittVector.teichmullerFun p r)) (WittVector.teichmul …
  -/
  ext n; cases n
    /-
      case h.zero
      p : Nat
      R : Type u_1
      S : Type u_2
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      r : R
      ⊢ Eq (((WittVector.map f) (WittVector.teichmullerFun p r)).coeff 0) ((WittVect …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      p : Nat
      R : Type u_1
      S : Type u_2
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      r : R
      n✝ : Nat
      ⊢ Eq (((WittVector.map f) (WittVector.teichmullerFun p r)).coeff (HAdd.hAdd n✝ …
    -/
  · exact f.map_zero
    /-
      🎉 no goals
    -/


private theorem teichmuller_mul_aux₁ {R : Type*} (x y : MvPolynomial R ℚ) :
    teichmullerFun p (x * y) = teichmullerFun p x * teichmullerFun p y := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    R : Type u_3
    x y : MvPolynomial R Rat
    ⊢ Eq (WittVector.teichmullerFun p (HMul.hMul x y)) (HMul.hMul (WittVector.teic …
  -/
  apply (ghostMap.bijective_of_invertible p (MvPolynomial R ℚ)).1
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    R : Type u_3
    x y : MvPolynomial R Rat
    ⊢ Eq (WittVector.ghostMap (WittVector.teichmullerFun p (HMul.hMul x y))) (Witt …
  -/
  rw [RingHom.map_mul]
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    R : Type u_3
    x y : MvPolynomial R Rat
    ⊢ Eq (WittVector.ghostMap (WittVector.teichmullerFun p (HMul.hMul x y))) (HMul …
  -/
  ext1 n
  /-
    case a.h
    p : Nat
    hp : Fact (Nat.Prime p)
    R : Type u_3
    x y : MvPolynomial R Rat
    n : Nat
    ⊢ Eq (WittVector.ghostMap (WittVector.teichmullerFun p (HMul.hMul x y)) n) (HM …
  -/
  simp only [Pi.mul_apply, ghostMap_apply, ghostComponent_teichmullerFun, mul_pow]
  /-
    🎉 no goals
  -/


private theorem teichmuller_mul_aux₂ {R : Type*} (x y : MvPolynomial R ℤ) :
    teichmullerFun p (x * y) = teichmullerFun p x * teichmullerFun p y := by
  refine map_injective (MvPolynomial.map (Int.castRingHom ℚ))
    (MvPolynomial.map_injective _ Int.cast_injective) ?_
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    R : Type u_3
    x y : MvPolynomial R Int
    ⊢ Eq ((WittVector.map (MvPolynomial.map (Int.castRingHom Rat))) (WittVector.te …
  -/
  simp only [teichmuller_mul_aux₁, map_teichmullerFun, RingHom.map_mul]
  /-
    🎉 no goals
  -/


/-- The Teichmüller lift of an element of `R` to `𝕎 R`.
The `0`-th coefficient of `teichmuller p r` is `r`, and all others are `0`.
This is a monoid homomorphism. -/
def teichmuller : R →* 𝕎 R where
  toFun := teichmullerFun p
  map_one' := by
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      ⊢ Eq (WittVector.teichmullerFun p 1) 1
    -/
    ext ⟨⟩
      /-
        case h.zero
        p : Nat
        R : Type u_1
        S : Type u_2
        hp : Fact (Nat.Prime p)
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        ⊢ Eq ((WittVector.teichmullerFun p 1).coeff 0) (WittVector.coeff 1 0)
      -/
    · rw [one_coeff_zero]; rfl
                           /-
                             🎉 no goals
                           -/
      /-
        case h.succ
        p : Nat
        R : Type u_1
        S : Type u_2
        hp : Fact (Nat.Prime p)
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        n✝ : Nat
        ⊢ Eq ((WittVector.teichmullerFun p 1).coeff (HAdd.hAdd n✝ 1)) (WittVector.coef …
      -/
    · rw [one_coeff_eq_of_pos _ _ _ (Nat.succ_pos _)]; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/
  map_mul' := by
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      ⊢ ∀ (x y : R), Eq ({ toFun := WittVector.teichmullerFun p, map_one' := ⋯ }.toF …
    -/
    intro x y
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x y : R
      ⊢ Eq ({ toFun := WittVector.teichmullerFun p, map_one' := ⋯ }.toFun (HMul.hMul …
    -/
    rcases counit_surjective R x with ⟨x, rfl⟩
    /-
      case intro
      p : Nat
      R : Type u_1
      S : Type u_2
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      y : R
      x : MvPolynomial R Int
      ⊢ Eq ({ toFun := WittVector.teichmullerFun p, map_one' := ⋯ }.toFun (HMul.hMul …
    -/
    rcases counit_surjective R y with ⟨y, rfl⟩
    /-
      case intro.intro
      p : Nat
      R : Type u_1
      S : Type u_2
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x y : MvPolynomial R Int
      ⊢ Eq ({ toFun := WittVector.teichmullerFun p, map_one' := ⋯ }.toFun (HMul.hMul …
    -/
    simp only [← map_teichmullerFun, ← RingHom.map_mul, teichmuller_mul_aux₂]
    /-
      🎉 no goals
    -/


@[simp]
theorem teichmuller_coeff_zero (r : R) : (teichmuller p r).coeff 0 = r :=
  rfl


@[simp]
theorem teichmuller_coeff_pos (r : R) : ∀ (n : ℕ) (_ : 0 < n), (teichmuller p r).coeff n = 0
  | _ + 1, _ => rfl


@[simp]
theorem teichmuller_zero : teichmuller p (0 : R) = 0 := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    ⊢ Eq ((WittVector.teichmuller p) 0) 0
  -/
                                /-
                                  🎉 no goals
                                -/
  ext ⟨⟩ <;> · rw [zero_coeff]; rfl
                                /-
                                  🎉 no goals
                                -/


/-- `teichmuller` is a natural transformation. -/
@[simp]
theorem map_teichmuller (f : R →+* S) (r : R) : map f (teichmuller p r) = teichmuller p (f r) :=
  map_teichmullerFun _ _ _


/-- The `n`-th ghost component of `teichmuller p r` is `r ^ p ^ n`. -/
@[simp]
theorem ghostComponent_teichmuller (r : R) (n : ℕ) :
    ghostComponent n (teichmuller p r) = r ^ p ^ n :=
  ghostComponent_teichmullerFun _ _ _


