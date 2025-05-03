local notation "𝕎" => WittVector p -- type as `\bbW`


/-- `verschiebungFun x` shifts the coefficients of `x` up by one,
by inserting 0 as the 0th coefficient.
`x.coeff i` then becomes `(verchiebungFun x).coeff (i + 1)`.

`verschiebungFun` is the underlying function of the additive monoid hom `WittVector.verschiebung`.
-/
def verschiebungFun (x : 𝕎 R) : 𝕎 R :=
  @mk' p _ fun n => if n = 0 then 0 else x.coeff (n - 1)


theorem verschiebungFun_coeff (x : 𝕎 R) (n : ℕ) :
    (verschiebungFun x).coeff n = if n = 0 then 0 else x.coeff (n - 1) := by
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    n : Nat
    ⊢ Eq (x.verschiebungFun.coeff n) (ite (Eq n 0) 0 (x.coeff (HSub.hSub n 1)))
  -/
  simp only [verschiebungFun]
  /-
    🎉 no goals
  -/


theorem verschiebungFun_coeff_zero (x : 𝕎 R) : (verschiebungFun x).coeff 0 = 0 := by
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    ⊢ Eq (x.verschiebungFun.coeff 0) 0
  -/
  rw [verschiebungFun_coeff, if_pos rfl]
  /-
    🎉 no goals
  -/


@[simp]
theorem verschiebungFun_coeff_succ (x : 𝕎 R) (n : ℕ) :
    (verschiebungFun x).coeff n.succ = x.coeff n :=
  rfl


@[ghost_simps]
theorem ghostComponent_zero_verschiebungFun [hp : Fact p.Prime] (x : 𝕎 R) :
    ghostComponent 0 (verschiebungFun x) = 0 := by
  rw [ghostComponent_apply, aeval_wittPolynomial, Finset.range_one, Finset.sum_singleton,
    verschiebungFun_coeff_zero, pow_zero, pow_zero, pow_one, one_mul]


@[ghost_simps]
theorem ghostComponent_verschiebungFun [hp : Fact p.Prime] (x : 𝕎 R) (n : ℕ) :
    ghostComponent (n + 1) (verschiebungFun x) = p * ghostComponent n x := by
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    x : WittVector p R
    n : Nat
    ⊢ Eq ((WittVector.ghostComponent (HAdd.hAdd n 1)) x.verschiebungFun) (HMul.hMu …
  -/
  simp only [ghostComponent_apply, aeval_wittPolynomial]
  rw [Finset.sum_range_succ', verschiebungFun_coeff, if_pos rfl,
    zero_pow (pow_ne_zero _ hp.1.ne_zero), mul_zero, add_zero, Finset.mul_sum, Finset.sum_congr rfl]
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    x : WittVector p R
    n : Nat
    ⊢ ∀ (x_1 : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x_1 → Eq (HMul. …
  -/
  rintro i -
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    x : WittVector p R
    n i : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (↑p) (HAdd.hAdd i 1)) (HPow.hPow (x.verschiebungFun …
  -/
  simp only [pow_succ', verschiebungFun_coeff_succ, Nat.succ_sub_succ_eq_sub, mul_assoc]
  /-
    🎉 no goals
  -/


/-- The 0th Verschiebung polynomial is 0. For `n > 0`, the `n`th Verschiebung polynomial is the
variable `X (n-1)`.
-/
def verschiebungPoly (n : ℕ) : MvPolynomial ℕ ℤ :=
  if n = 0 then 0 else X (n - 1)


@[simp]
theorem verschiebungPoly_zero : verschiebungPoly 0 = 0 :=
  rfl


theorem aeval_verschiebung_poly' (x : 𝕎 R) (n : ℕ) :
    aeval x.coeff (verschiebungPoly n) = (verschiebungFun x).coeff n := by
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    n : Nat
    ⊢ Eq ((MvPolynomial.aeval x.coeff) (WittVector.verschiebungPoly n)) (x.verschi …
  -/
  cases' n with n
    /-
      case zero
      p : Nat
      R : Type u_1
      inst✝ : CommRing R
      x : WittVector p R
      ⊢ Eq ((MvPolynomial.aeval x.coeff) (WittVector.verschiebungPoly 0)) (x.verschi …
    -/
  · simp only [verschiebungPoly, ite_true, map_zero, verschiebungFun_coeff_zero]
    /-
      🎉 no goals
    -/
  · rw [verschiebungPoly, verschiebungFun_coeff_succ, if_neg n.succ_ne_zero, aeval_X,
      add_tsub_cancel_right]


/-- `WittVector.verschiebung` has polynomial structure given by `WittVector.verschiebungPoly`.
-/
-- Porting note: replaced `@[is_poly]` with `instance`.
instance verschiebungFun_isPoly : IsPoly p fun R _Rcr => @verschiebungFun p R _Rcr := by
  /-
    p : Nat
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    ⊢ WittVector.IsPoly p fun R _Rcr => WittVector.verschiebungFun
  -/
  use verschiebungPoly
  /-
    case h
    p : Nat
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    ⊢ ∀ ⦃R : Type u_3⦄ [inst : CommRing R] (x : WittVector p R), Eq x.verschiebung …
  -/
  simp only [aeval_verschiebung_poly', eq_self_iff_true, forall₃_true_iff]
  /-
    🎉 no goals
  -/

-- Porting note: we add this example as a verification that Lean 4's instance resolution
-- can handle what in Lean 3 we needed the `@[is_poly]` attribute to help with.

/--
`verschiebung x` shifts the coefficients of `x` up by one, by inserting 0 as the 0th coefficient.
`x.coeff i` then becomes `(verchiebung x).coeff (i + 1)`.

This is an additive monoid hom with underlying function `verschiebung_fun`.
-/
noncomputable def verschiebung : 𝕎 R →+ 𝕎 R where
  toFun := verschiebungFun
  map_zero' := by
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      hp : Fact (Nat.Prime p)
      ⊢ Eq (WittVector.verschiebungFun 0) 0
    -/
    ext ⟨⟩ <;> rw [verschiebungFun_coeff] <;>
      /-
        case h.zero
        p : Nat
        R : Type u_1
        S : Type u_2
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        hp : Fact (Nat.Prime p)
        ⊢ Eq (ite (Eq 0 0) 0 (WittVector.coeff 0 (HSub.hSub 0 1))) (WittVector.coeff 0 …
      -/
      /-
        🎉 no goals
      -/
      simp only [if_true, eq_self_iff_true, zero_coeff, ite_self]
      /-
        🎉 no goals
      -/
  map_add' := by
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      hp : Fact (Nat.Prime p)
      ⊢ ∀ (x y : WittVector p R), Eq ({ toFun := WittVector.verschiebungFun, map_zer …
    -/
    dsimp
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      hp : Fact (Nat.Prime p)
      ⊢ ∀ (x y : WittVector p R), Eq (HAdd.hAdd x y).verschiebungFun (HAdd.hAdd x.ve …
    -/
    ghost_calc _ _
    /-
      case refine_3
      p : Nat
      S : Type u_2
      inst✝ : CommRing S
      hp : Fact (Nat.Prime p)
      R : Type u_1
      R._inst : CommRing R
      x✝ y✝ : WittVector p R
      ⊢ ∀ (n : Nat), Eq ((WittVector.ghostComponent n) (HAdd.hAdd x✝ y✝).verschiebun …
    -/
                  /-
                    🎉 no goals
                  -/
    rintro ⟨⟩ <;> ghost_simp
                  /-
                    🎉 no goals
                  -/


/-- `WittVector.verschiebung` is a polynomial function. -/
@[is_poly]
theorem verschiebung_isPoly : IsPoly p fun _ _ => verschiebung (p := p) :=
  verschiebungFun_isPoly p


/-- verschiebung is a natural transformation -/
@[simp]
theorem map_verschiebung (f : R →+* S) (x : 𝕎 R) :
    map f (verschiebung x) = verschiebung (map f x) := by
  /-
    p : Nat
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    hp : Fact (Nat.Prime p)
    f : RingHom R S
    x : WittVector p R
    ⊢ Eq ((WittVector.map f) (WittVector.verschiebung x)) (WittVector.verschiebung …
  -/
  ext ⟨-, -⟩
    /-
      case h.zero
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      hp : Fact (Nat.Prime p)
      f : RingHom R S
      x : WittVector p R
      ⊢ Eq (((WittVector.map f) (WittVector.verschiebung x)).coeff 0) ((WittVector.v …
    -/
  · exact f.map_zero
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      hp : Fact (Nat.Prime p)
      f : RingHom R S
      x : WittVector p R
      n✝ : Nat
      ⊢ Eq (((WittVector.map f) (WittVector.verschiebung x)).coeff (HAdd.hAdd n✝ 1)) …
    -/
  · rfl
    /-
      🎉 no goals
    -/


@[ghost_simps]
theorem ghostComponent_zero_verschiebung (x : 𝕎 R) : ghostComponent 0 (verschiebung x) = 0 :=
  ghostComponent_zero_verschiebungFun _


@[ghost_simps]
theorem ghostComponent_verschiebung (x : 𝕎 R) (n : ℕ) :
    ghostComponent (n + 1) (verschiebung x) = p * ghostComponent n x :=
  ghostComponent_verschiebungFun _ _


@[simp]
theorem verschiebung_coeff_zero (x : 𝕎 R) : (verschiebung x).coeff 0 = 0 :=
  rfl

-- simp_nf complains if this is simp

theorem verschiebung_coeff_add_one (x : 𝕎 R) (n : ℕ) :
    (verschiebung x).coeff (n + 1) = x.coeff n :=
  rfl


@[simp]
theorem verschiebung_coeff_succ (x : 𝕎 R) (n : ℕ) : (verschiebung x).coeff n.succ = x.coeff n :=
  rfl


theorem aeval_verschiebungPoly (x : 𝕎 R) (n : ℕ) :
    aeval x.coeff (verschiebungPoly n) = (verschiebung x).coeff n :=
  aeval_verschiebung_poly' x n


@[simp]
theorem bind₁_verschiebungPoly_wittPolynomial (n : ℕ) :
    bind₁ verschiebungPoly (wittPolynomial p ℤ n) =
      if n = 0 then 0 else p * wittPolynomial p ℤ (n - 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((MvPolynomial.bind₁ WittVector.verschiebungPoly) (wittPolynomial p Int n …
  -/
  apply MvPolynomial.funext
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ ∀ (x : Nat → Int), Eq ((MvPolynomial.eval x) ((MvPolynomial.bind₁ WittVector …
  -/
  intro x
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    x : Nat → Int
    ⊢ Eq ((MvPolynomial.eval x) ((MvPolynomial.bind₁ WittVector.verschiebungPoly)  …
  -/
  split_ifs with hn
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      x : Nat → Int
      hn : Eq n 0
      ⊢ Eq ((MvPolynomial.eval x) ((MvPolynomial.bind₁ WittVector.verschiebungPoly)  …
    -/
  · simp only [hn, wittPolynomial_zero, bind₁_X_right, verschiebungPoly_zero, map_zero, ite_true]
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      x : Nat → Int
      hn : Not (Eq n 0)
      ⊢ Eq ((MvPolynomial.eval x) ((MvPolynomial.bind₁ WittVector.verschiebungPoly)  …
    -/
  · obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hn
    /-
      case neg.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Nat → Int
      n : Nat
      hn : Not (Eq n.succ 0)
      ⊢ Eq ((MvPolynomial.eval x) ((MvPolynomial.bind₁ WittVector.verschiebungPoly)  …
    -/
    rw [Nat.succ_eq_add_one, add_tsub_cancel_right]
    /-
      case neg.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Nat → Int
      n : Nat
      hn : Not (Eq n.succ 0)
      ⊢ Eq ((MvPolynomial.eval x) ((MvPolynomial.bind₁ WittVector.verschiebungPoly)  …
    -/
    simp only [add_eq_zero, and_false, ite_false, map_mul]
    /-
      case neg.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Nat → Int
      n : Nat
      hn : Not (Eq n.succ 0)
      ⊢ Eq ((MvPolynomial.eval x) ((MvPolynomial.bind₁ WittVector.verschiebungPoly)  …
    -/
    rw [map_natCast, hom_bind₁]
    calc
      _ = ghostComponent (n + 1) (verschiebung <| mk p x) := by
       apply eval₂Hom_congr (RingHom.ext_int _ _) _ rfl
       funext k
       simp only [← aeval_verschiebungPoly]
       exact eval₂Hom_congr (RingHom.ext_int _ _) rfl rfl
      _ = _ := by rw [ghostComponent_verschiebung]; rfl


