local notation "𝕎" => WittVector p

local notation "W_" => wittPolynomial p

-- type as `\bbW`

/-- `f : α → β` induces a map from `𝕎 α` to `𝕎 β` by applying `f` componentwise.
If `f` is a ring homomorphism, then so is `f`, see `WittVector.map f`. -/
def mapFun (f : α → β) : 𝕎 α → 𝕎 β := fun x => mk _ (f ∘ x.coeff)


theorem injective (f : α → β) (hf : Injective f) : Injective (mapFun f : 𝕎 α → 𝕎 β) := by
  /-
    p : Nat
    α : Type u_3
    β : Type u_4
    f : α → β
    hf : Function.Injective f
    ⊢ Function.Injective (WittVector.mapFun f)
  -/
  intros _ _ h
  /-
    p : Nat
    α : Type u_3
    β : Type u_4
    f : α → β
    hf : Function.Injective f
    a₁✝ a₂✝ : WittVector p α
    h : Eq (WittVector.mapFun f a₁✝) (WittVector.mapFun f a₂✝)
    ⊢ Eq a₁✝ a₂✝
  -/
  ext p
  /-
    case h
    p✝ : Nat
    α : Type u_3
    β : Type u_4
    f : α → β
    hf : Function.Injective f
    a₁✝ a₂✝ : WittVector p✝ α
    h : Eq (WittVector.mapFun f a₁✝) (WittVector.mapFun f a₂✝)
    p : Nat
    ⊢ Eq (a₁✝.coeff p) (a₂✝.coeff p)
  -/
  exact hf (congr_arg (fun x => coeff x p) h : _)
  /-
    🎉 no goals
  -/


theorem surjective (f : α → β) (hf : Surjective f) : Surjective (mapFun f : 𝕎 α → 𝕎 β) := fun x =>
  ⟨mk _ fun n => Classical.choose <| hf <| x.coeff n,
       /-
         p : Nat
         α : Type u_3
         β : Type u_4
         f : α → β
         hf : Function.Surjective f
         x : WittVector p β
         ⊢ Eq (WittVector.mapFun f (WittVector.mk p fun n => Classical.choose ⋯)) x
       -/
    by ext n; simp only [mapFun, coeff_mk, comp_apply, Classical.choose_spec (hf (x.coeff n))]⟩
              /-
                🎉 no goals
              -/


/-- Auxiliary tactic for showing that `mapFun` respects the ring operations. -/
-- porting note: a very crude port.
macro "map_fun_tac" : tactic => `(tactic| (
  ext n
  simp only [mapFun, mk, comp_apply, zero_coeff, map_zero,
    -- Porting note: the lemmas on the next line do not have the `simp` tag in mathlib4
    add_coeff, sub_coeff, mul_coeff, neg_coeff, nsmul_coeff, zsmul_coeff, pow_coeff,
    peval, map_aeval, algebraMap_int_eq, coe_eval₂Hom] <;>
  try { cases n <;> simp <;> done } <;>  -- Porting note: this line solves `one`
  apply eval₂Hom_congr (RingHom.ext_int _ _) _ rfl <;>
  ext ⟨i, k⟩ <;>
    fin_cases i <;> rfl))


                                            /-
                                              p : Nat
                                              R : Type u_1
                                              S : Type u_2
                                              inst✝² : CommRing R
                                              inst✝¹ : CommRing S
                                              inst✝ : Fact (Nat.Prime p)
                                              f : RingHom R S
                                              ⊢ Eq (WittVector.mapFun (⇑f) 0) 0
                                            -/
theorem zero : mapFun f (0 : 𝕎 R) = 0 := by map_fun_tac
                                            /-
                                              🎉 no goals
                                            -/


                                           /-
                                             p : Nat
                                             R : Type u_1
                                             S : Type u_2
                                             inst✝² : CommRing R
                                             inst✝¹ : CommRing S
                                             inst✝ : Fact (Nat.Prime p)
                                             f : RingHom R S
                                             ⊢ Eq (WittVector.mapFun (⇑f) 1) 1
                                           -/
theorem one : mapFun f (1 : 𝕎 R) = 1 := by map_fun_tac
                                           /-
                                             🎉 no goals
                                           -/


                                                               /-
                                                                 p : Nat
                                                                 R : Type u_1
                                                                 S : Type u_2
                                                                 inst✝² : CommRing R
                                                                 inst✝¹ : CommRing S
                                                                 inst✝ : Fact (Nat.Prime p)
                                                                 f : RingHom R S
                                                                 x y : WittVector p R
                                                                 ⊢ Eq (WittVector.mapFun (⇑f) (HAdd.hAdd x y)) (HAdd.hAdd (WittVector.mapFun (⇑ …
                                                               -/
theorem add : mapFun f (x + y) = mapFun f x + mapFun f y := by map_fun_tac
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                               /-
                                                                 p : Nat
                                                                 R : Type u_1
                                                                 S : Type u_2
                                                                 inst✝² : CommRing R
                                                                 inst✝¹ : CommRing S
                                                                 inst✝ : Fact (Nat.Prime p)
                                                                 f : RingHom R S
                                                                 x y : WittVector p R
                                                                 ⊢ Eq (WittVector.mapFun (⇑f) (HSub.hSub x y)) (HSub.hSub (WittVector.mapFun (⇑ …
                                                               -/
theorem sub : mapFun f (x - y) = mapFun f x - mapFun f y := by map_fun_tac
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                               /-
                                                                 p : Nat
                                                                 R : Type u_1
                                                                 S : Type u_2
                                                                 inst✝² : CommRing R
                                                                 inst✝¹ : CommRing S
                                                                 inst✝ : Fact (Nat.Prime p)
                                                                 f : RingHom R S
                                                                 x y : WittVector p R
                                                                 ⊢ Eq (WittVector.mapFun (⇑f) (HMul.hMul x y)) (HMul.hMul (WittVector.mapFun (⇑ …
                                                               -/
theorem mul : mapFun f (x * y) = mapFun f x * mapFun f y := by map_fun_tac
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                /-
                                                  p : Nat
                                                  R : Type u_1
                                                  S : Type u_2
                                                  inst✝² : CommRing R
                                                  inst✝¹ : CommRing S
                                                  inst✝ : Fact (Nat.Prime p)
                                                  f : RingHom R S
                                                  x : WittVector p R
                                                  ⊢ Eq (WittVector.mapFun (⇑f) (Neg.neg x)) (Neg.neg (WittVector.mapFun (⇑f) x))
                                                -/
theorem neg : mapFun f (-x) = -mapFun f x := by map_fun_tac
                                                /-
                                                  🎉 no goals
                                                -/


                                                                                     /-
                                                                                       p : Nat
                                                                                       R : Type u_1
                                                                                       S : Type u_2
                                                                                       inst✝² : CommRing R
                                                                                       inst✝¹ : CommRing S
                                                                                       inst✝ : Fact (Nat.Prime p)
                                                                                       f : RingHom R S
                                                                                       n : Nat
                                                                                       x : WittVector p R
                                                                                       ⊢ Eq (WittVector.mapFun (⇑f) (HSMul.hSMul n x)) (HSMul.hSMul n (WittVector.map …
                                                                                     -/
theorem nsmul (n : ℕ) (x : WittVector p R) : mapFun f (n • x) = n • mapFun f x := by map_fun_tac
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


                                                                                     /-
                                                                                       p : Nat
                                                                                       R : Type u_1
                                                                                       S : Type u_2
                                                                                       inst✝² : CommRing R
                                                                                       inst✝¹ : CommRing S
                                                                                       inst✝ : Fact (Nat.Prime p)
                                                                                       f : RingHom R S
                                                                                       z : Int
                                                                                       x : WittVector p R
                                                                                       ⊢ Eq (WittVector.mapFun (⇑f) (HSMul.hSMul z x)) (HSMul.hSMul z (WittVector.map …
                                                                                     -/
theorem zsmul (z : ℤ) (x : WittVector p R) : mapFun f (z • x) = z • mapFun f x := by map_fun_tac
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


                                                              /-
                                                                p : Nat
                                                                R : Type u_1
                                                                S : Type u_2
                                                                inst✝² : CommRing R
                                                                inst✝¹ : CommRing S
                                                                inst✝ : Fact (Nat.Prime p)
                                                                f : RingHom R S
                                                                x : WittVector p R
                                                                n : Nat
                                                                ⊢ Eq (WittVector.mapFun (⇑f) (HPow.hPow x n)) (HPow.hPow (WittVector.mapFun (⇑ …
                                                              -/
theorem pow (n : ℕ) : mapFun f (x ^ n) = mapFun f x ^ n := by map_fun_tac
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem natCast (n : ℕ) : mapFun f (n : 𝕎 R) = n :=
  show mapFun f n.unaryCast = (n : WittVector p S) by
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Fact (Nat.Prime p)
      f : RingHom R S
      n : Nat
      ⊢ Eq (WittVector.mapFun (⇑f) n.unaryCast) ↑n
    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    induction n <;> simp [*, Nat.unaryCast, add, one, zero] <;> rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[deprecated (since := "2024-04-17")]
alias nat_cast := natCast


theorem intCast (n : ℤ) : mapFun f (n : 𝕎 R) = n :=
  show mapFun f n.castDef = (n : WittVector p S) by
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Fact (Nat.Prime p)
      f : RingHom R S
      n : Int
      ⊢ Eq (WittVector.mapFun (⇑f) n.castDef) ↑n
    -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    cases n <;> simp [*, Int.castDef, add, one, neg, zero, natCast] <;> rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[deprecated (since := "2024-04-17")]
alias int_cast := intCast


/-- Evaluates the `n`th Witt polynomial on the first `n` coefficients of `x`,
producing a value in `R`.
This function will be bundled as the ring homomorphism `WittVector.ghostMap`
once the ring structure is available,
but we rely on it to set up the ring structure in the first place. -/
private def ghostFun : 𝕎 R → ℕ → R := fun x n => aeval x.coeff (W_ ℤ n)


/-- An auxiliary tactic for proving that `ghostFun` respects the ring operations. -/
elab "ghost_fun_tac" φ:term "," fn:term : tactic => do
  evalTactic (← `(tactic| (
  ext n
  have := congr_fun (congr_arg (@peval R _ _) (wittStructureInt_prop p $φ n)) $fn
  simp only [wittZero, OfNat.ofNat, Zero.zero, wittOne, One.one,
    HAdd.hAdd, Add.add, HSub.hSub, Sub.sub, Neg.neg, HMul.hMul, Mul.mul,HPow.hPow, Pow.pow,
    wittNSMul, wittZSMul, HSMul.hSMul, SMul.smul]
  simpa (config := { unfoldPartialApp := true }) [WittVector.ghostFun, aeval_rename, aeval_bind₁,
    comp, uncurry, peval, eval] using this
  )))


@[local simp]
theorem matrix_vecEmpty_coeff {R} (i j) :
    @coeff p R (Matrix.vecEmpty i) j = (Matrix.vecEmpty i : ℕ → R) j := by
  /-
    p : Nat
    R : Type u_5
    i : Fin 0
    j : Nat
    ⊢ Eq ((Matrix.vecEmpty i).coeff j) (Matrix.vecEmpty i j)
  -/
  rcases i with ⟨_ | _ | _ | _ | i_val, ⟨⟩⟩
  /-
    🎉 no goals
  -/


private theorem ghostFun_zero : ghostFun (0 : 𝕎 R) = 0 := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    ⊢ Eq (WittVector.ghostFun 0) 0
  -/
  ghost_fun_tac 0, ![]
  /-
    🎉 no goals
  -/


private theorem ghostFun_one : ghostFun (1 : 𝕎 R) = 1 := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    ⊢ Eq (WittVector.ghostFun 1) 1
  -/
  ghost_fun_tac 1, ![]
  /-
    🎉 no goals
  -/


private theorem ghostFun_add : ghostFun (x + y) = ghostFun x + ghostFun y := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x y : WittVector p R
    ⊢ Eq (WittVector.ghostFun (HAdd.hAdd x y)) (HAdd.hAdd (WittVector.ghostFun x)  …
  -/
  ghost_fun_tac X 0 + X 1, ![x.coeff, y.coeff]
  /-
    🎉 no goals
  -/


private theorem ghostFun_natCast (i : ℕ) : ghostFun (i : 𝕎 R) = i :=
  show ghostFun i.unaryCast = _ by
    /-
      p : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : Fact (Nat.Prime p)
      i : Nat
      ⊢ Eq (WittVector.ghostFun i.unaryCast) ↑i
    -/
    induction i <;>
      /-
        case zero
        p : Nat
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : Fact (Nat.Prime p)
        ⊢ Eq (WittVector.ghostFun (Nat.unaryCast 0)) ↑0
      -/
      /-
        🎉 no goals
      -/
      simp [*, Nat.unaryCast, ghostFun_zero, ghostFun_one, ghostFun_add, -Pi.natCast_def]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-04-17")]
alias ghostFun_nat_cast := ghostFun_natCast


private theorem ghostFun_sub : ghostFun (x - y) = ghostFun x - ghostFun y := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x y : WittVector p R
    ⊢ Eq (WittVector.ghostFun (HSub.hSub x y)) (HSub.hSub (WittVector.ghostFun x)  …
  -/
  ghost_fun_tac X 0 - X 1, ![x.coeff, y.coeff]
  /-
    🎉 no goals
  -/


private theorem ghostFun_mul : ghostFun (x * y) = ghostFun x * ghostFun y := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x y : WittVector p R
    ⊢ Eq (WittVector.ghostFun (HMul.hMul x y)) (HMul.hMul (WittVector.ghostFun x)  …
  -/
  ghost_fun_tac X 0 * X 1, ![x.coeff, y.coeff]
  /-
    🎉 no goals
  -/


                                                                 /-
                                                                   p : Nat
                                                                   R : Type u_1
                                                                   inst✝¹ : CommRing R
                                                                   inst✝ : Fact (Nat.Prime p)
                                                                   x : WittVector p R
                                                                   ⊢ Eq (WittVector.ghostFun (Neg.neg x)) (Neg.neg (WittVector.ghostFun x))
                                                                 -/
private theorem ghostFun_neg : ghostFun (-x) = -ghostFun x := by ghost_fun_tac -X 0, ![x.coeff]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


private theorem ghostFun_intCast (i : ℤ) : ghostFun (i : 𝕎 R) = i :=
  show ghostFun i.castDef = _ by
    /-
      p : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : Fact (Nat.Prime p)
      i : Int
      ⊢ Eq (WittVector.ghostFun i.castDef) ↑i
    -/
    cases i <;> simp [*, Int.castDef, ghostFun_natCast, ghostFun_neg, -Pi.natCast_def,
      -Pi.intCast_def]


@[deprecated (since := "2024-04-17")]
alias ghostFun_int_cast := ghostFun_intCast


private lemma ghostFun_nsmul (m : ℕ) (x : WittVector p R) : ghostFun (m • x) = m • ghostFun x := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Nat
    x : WittVector p R
    ⊢ Eq (WittVector.ghostFun (HSMul.hSMul m x)) (HSMul.hSMul m (WittVector.ghostF …
  -/
  ghost_fun_tac m • (X 0), ![x.coeff]
  /-
    🎉 no goals
  -/


private lemma ghostFun_zsmul (m : ℤ) (x : WittVector p R) : ghostFun (m • x) = m • ghostFun x := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Int
    x : WittVector p R
    ⊢ Eq (WittVector.ghostFun (HSMul.hSMul m x)) (HSMul.hSMul m (WittVector.ghostF …
  -/
  ghost_fun_tac m • (X 0), ![x.coeff]
  /-
    🎉 no goals
  -/


private theorem ghostFun_pow (m : ℕ) : ghostFun (x ^ m) = ghostFun x ^ m := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x : WittVector p R
    m : Nat
    ⊢ Eq (WittVector.ghostFun (HPow.hPow x m)) (HPow.hPow (WittVector.ghostFun x) m)
  -/
  ghost_fun_tac X 0 ^ m, ![x.coeff]
  /-
    🎉 no goals
  -/


/-- The bijection between `𝕎 R` and `ℕ → R`, under the assumption that `p` is invertible in `R`.
In `WittVector.ghostEquiv` we upgrade this to an isomorphism of rings. -/
private def ghostEquiv' [Invertible (p : R)] : 𝕎 R ≃ (ℕ → R) where
  toFun := ghostFun
  invFun x := mk p fun n => aeval x (xInTermsOfW p R n)
  left_inv := by
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      α : Type u_3
      β : Type u_4
      inst✝ : Invertible ↑p
      ⊢ Function.LeftInverse (fun x => WittVector.mk p fun n => (MvPolynomial.aeval  …
    -/
    intro x
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      α : Type u_3
      β : Type u_4
      inst✝ : Invertible ↑p
      x : WittVector p R
      ⊢ Eq ((fun x => WittVector.mk p fun n => (MvPolynomial.aeval x) (xInTermsOfW p …
    -/
    ext n
    /-
      case h
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      α : Type u_3
      β : Type u_4
      inst✝ : Invertible ↑p
      x : WittVector p R
      n : Nat
      ⊢ Eq (((fun x => WittVector.mk p fun n => (MvPolynomial.aeval x) (xInTermsOfW  …
    -/
    have := bind₁_wittPolynomial_xInTermsOfW p R n
    /-
      case h
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      α : Type u_3
      β : Type u_4
      inst✝ : Invertible ↑p
      x : WittVector p R
      n : Nat
      this : Eq ((MvPolynomial.bind₁ (wittPolynomial p R)) (xInTermsOfW p R n)) (MvP …
      ⊢ Eq (((fun x => WittVector.mk p fun n => (MvPolynomial.aeval x) (xInTermsOfW  …
    -/
    apply_fun aeval x.coeff at this
    simpa (config := { unfoldPartialApp := true }) only [aeval_bind₁, aeval_X, ghostFun,
      aeval_wittPolynomial]
  right_inv := by
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      α : Type u_3
      β : Type u_4
      inst✝ : Invertible ↑p
      ⊢ Function.RightInverse (fun x => WittVector.mk p fun n => (MvPolynomial.aeval …
    -/
    intro x
    /-
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      α : Type u_3
      β : Type u_4
      inst✝ : Invertible ↑p
      x : Nat → R
      ⊢ Eq (WittVector.ghostFun ((fun x => WittVector.mk p fun n => (MvPolynomial.ae …
    -/
    ext n
    /-
      case h
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      α : Type u_3
      β : Type u_4
      inst✝ : Invertible ↑p
      x : Nat → R
      n : Nat
      ⊢ Eq (WittVector.ghostFun ((fun x => WittVector.mk p fun n => (MvPolynomial.ae …
    -/
    have := bind₁_xInTermsOfW_wittPolynomial p R n
    /-
      case h
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      α : Type u_3
      β : Type u_4
      inst✝ : Invertible ↑p
      x : Nat → R
      n : Nat
      this : Eq ((MvPolynomial.bind₁ (xInTermsOfW p R)) (wittPolynomial p R n)) (MvP …
      ⊢ Eq (WittVector.ghostFun ((fun x => WittVector.mk p fun n => (MvPolynomial.ae …
    -/
    apply_fun aeval x at this
    /-
      case h
      p : Nat
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      α : Type u_3
      β : Type u_4
      inst✝ : Invertible ↑p
      x : Nat → R
      n : Nat
      this : Eq ((MvPolynomial.aeval x) ((MvPolynomial.bind₁ (xInTermsOfW p R)) (wit …
      ⊢ Eq (WittVector.ghostFun ((fun x => WittVector.mk p fun n => (MvPolynomial.ae …
    -/
    simpa only [aeval_bind₁, aeval_X, ghostFun, aeval_wittPolynomial]
    /-
      🎉 no goals
    -/


@[local instance]
private def comm_ring_aux₁ : CommRing (𝕎 (MvPolynomial R ℚ)) :=
  (ghostEquiv' p (MvPolynomial R ℚ)).injective.commRing ghostFun ghostFun_zero ghostFun_one
    ghostFun_add ghostFun_mul ghostFun_neg ghostFun_sub ghostFun_nsmul ghostFun_zsmul
    ghostFun_pow ghostFun_natCast ghostFun_intCast


@[local instance]
private abbrev comm_ring_aux₂ : CommRing (𝕎 (MvPolynomial R ℤ)) :=
  (mapFun.injective _ <| map_injective (Int.castRingHom ℚ) Int.cast_injective).commRing _
    (mapFun.zero _) (mapFun.one _) (mapFun.add _) (mapFun.mul _) (mapFun.neg _) (mapFun.sub _)
    (mapFun.nsmul _) (mapFun.zsmul _) (mapFun.pow _) (mapFun.natCast _) (mapFun.intCast _)


/-- The commutative ring structure on `𝕎 R`. -/
instance : CommRing (𝕎 R) :=
  (mapFun.surjective _ <| counit_surjective _).commRing (mapFun <| MvPolynomial.counit _)
    (mapFun.zero _) (mapFun.one _) (mapFun.add _) (mapFun.mul _) (mapFun.neg _) (mapFun.sub _)
    (mapFun.nsmul _) (mapFun.zsmul _) (mapFun.pow _) (mapFun.natCast _) (mapFun.intCast _)


/-- `WittVector.map f` is the ring homomorphism `𝕎 R →+* 𝕎 S` naturally induced
by a ring homomorphism `f : R →+* S`. It acts coefficientwise. -/
noncomputable def map (f : R →+* S) : 𝕎 R →+* 𝕎 S where
  toFun := mapFun f
  map_zero' := mapFun.zero f
  map_one' := mapFun.one f
  map_add' := mapFun.add f
  map_mul' := mapFun.mul f


theorem map_injective (f : R →+* S) (hf : Injective f) : Injective (map f : 𝕎 R → 𝕎 S) :=
  mapFun.injective f hf


theorem map_surjective (f : R →+* S) (hf : Surjective f) : Surjective (map f : 𝕎 R → 𝕎 S) :=
  mapFun.surjective f hf


@[simp]
theorem map_coeff (f : R →+* S) (x : 𝕎 R) (n : ℕ) : (map f x).coeff n = f (x.coeff n) :=
  rfl


/-- `WittVector.ghostMap` is a ring homomorphism that maps each Witt vector
to the sequence of its ghost components. -/
def ghostMap : 𝕎 R →+* ℕ → R where
  toFun := ghostFun
  map_zero' := ghostFun_zero
  map_one' := ghostFun_one
  map_add' := ghostFun_add
  map_mul' := ghostFun_mul


/-- Evaluates the `n`th Witt polynomial on the first `n` coefficients of `x`,
producing a value in `R`. -/
def ghostComponent (n : ℕ) : 𝕎 R →+* R :=
  (Pi.evalRingHom _ n).comp ghostMap


theorem ghostComponent_apply (n : ℕ) (x : 𝕎 R) : ghostComponent n x = aeval x.coeff (W_ ℤ n) :=
  rfl


@[simp]
theorem ghostMap_apply (x : 𝕎 R) (n : ℕ) : ghostMap x n = ghostComponent n x :=
  rfl


/-- `WittVector.ghostMap` is a ring isomorphism when `p` is invertible in `R`. -/
def ghostEquiv : 𝕎 R ≃+* (ℕ → R) :=
  { (ghostMap : 𝕎 R →+* ℕ → R), ghostEquiv' p R with }


@[simp]
theorem ghostEquiv_coe : (ghostEquiv p R : 𝕎 R →+* ℕ → R) = ghostMap :=
  rfl


theorem ghostMap.bijective_of_invertible : Function.Bijective (ghostMap : 𝕎 R → ℕ → R) :=
  (ghostEquiv p R).bijective


/-- `WittVector.coeff x 0` as a `RingHom` -/
@[simps]
noncomputable def constantCoeff : 𝕎 R →+* R where
  toFun x := x.coeff 0
                  /-
                    p : Nat
                    R : Type u_1
                    S : Type u_2
                    inst✝² : CommRing R
                    inst✝¹ : CommRing S
                    α : Type u_3
                    β : Type u_4
                    inst✝ : Fact (Nat.Prime p)
                    ⊢ Eq ((↑{ toFun := fun x => x.coeff 0, map_one' := ⋯, map_mul' := ⋯ }).toFun 0 …
                  -/
                 /-
                   p : Nat
                   R : Type u_1
                   S : Type u_2
                   inst✝² : CommRing R
                   inst✝¹ : CommRing S
                   α : Type u_3
                   β : Type u_4
                   inst✝ : Fact (Nat.Prime p)
                   ⊢ Eq ((fun x => x.coeff 0) 1) 1
                 -/
  map_zero' := by simp
                 /-
                   🎉 no goals
                 -/
                  /-
                    🎉 no goals
                  -/
  map_one' := by simp
  map_add' := add_coeff_zero
  map_mul' := mul_coeff_zero


instance [Nontrivial R] : Nontrivial (𝕎 R) :=
  constantCoeff.domain_nontrivial


