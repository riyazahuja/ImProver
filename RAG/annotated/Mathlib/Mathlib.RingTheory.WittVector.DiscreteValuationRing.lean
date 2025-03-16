local notation "𝕎" => WittVector p


/-- This is the `n+1`st coefficient of our inverse. -/
def succNthValUnits (n : ℕ) (a : Units k) (A : 𝕎 k) (bs : Fin (n + 1) → k) : k :=
  -↑(a⁻¹ ^ p ^ (n + 1)) *
    (A.coeff (n + 1) * ↑(a⁻¹ ^ p ^ (n + 1)) + nthRemainder p n (truncateFun (n + 1) A) bs)


/--
Recursively defines the sequence of coefficients for the inverse to a Witt vector whose first entry
is a unit.
-/
noncomputable def inverseCoeff (a : Units k) (A : 𝕎 k) : ℕ → k
  | 0 => ↑a⁻¹
  | n + 1 => succNthValUnits n a A fun i => inverseCoeff a A i.val


/--
Upgrade a Witt vector `A` whose first entry `A.coeff 0` is a unit to be, itself, a unit in `𝕎 k`.
-/
def mkUnit {a : Units k} {A : 𝕎 k} (hA : A.coeff 0 = a) : Units (𝕎 k) :=
  Units.mkOfMulEqOne A (@WittVector.mk' p _ (inverseCoeff a A)) (by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝¹ : CommRing k
      inst✝ : CharP k p
      a : Units k
      A : WittVector p k
      hA : Eq (A.coeff 0) ↑a
      ⊢ Eq (HMul.hMul A { coeff := WittVector.inverseCoeff a A }) 1
    -/
    ext n
    /-
      case h
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝¹ : CommRing k
      inst✝ : CharP k p
      a : Units k
      A : WittVector p k
      hA : Eq (A.coeff 0) ↑a
      n : Nat
      ⊢ Eq ((HMul.hMul A { coeff := WittVector.inverseCoeff a A }).coeff n) (WittVec …
    -/
    induction' n with n _
      /-
        case h.zero
        p : Nat
        hp : Fact (Nat.Prime p)
        k : Type u_1
        inst✝¹ : CommRing k
        inst✝ : CharP k p
        a : Units k
        A : WittVector p k
        hA : Eq (A.coeff 0) ↑a
        ⊢ Eq ((HMul.hMul A { coeff := WittVector.inverseCoeff a A }).coeff 0) (WittVec …
      -/
    · simp [WittVector.mul_coeff_zero, inverseCoeff, hA]
      /-
        🎉 no goals
      -/
    let H_coeff := A.coeff (n + 1) * ↑(a⁻¹ ^ p ^ (n + 1)) +
      nthRemainder p n (truncateFun (n + 1) A) fun i : Fin (n + 1) => inverseCoeff a A i
    /-
      case h.succ
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝¹ : CommRing k
      inst✝ : CharP k p
      a : Units k
      A : WittVector p k
      hA : Eq (A.coeff 0) ↑a
      n : Nat
      a✝ : Eq ((HMul.hMul A { coeff := WittVector.inverseCoeff a A }).coeff n) (Witt …
      H_coeff : k := HAdd.hAdd (HMul.hMul (A.coeff (HAdd.hAdd n 1)) ↑(HPow.hPow (Inv …
      ⊢ Eq ((HMul.hMul A { coeff := WittVector.inverseCoeff a A }).coeff (HAdd.hAdd  …
    -/
    have H := Units.mul_inv (a ^ p ^ (n + 1))
    /-
      case h.succ
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝¹ : CommRing k
      inst✝ : CharP k p
      a : Units k
      A : WittVector p k
      hA : Eq (A.coeff 0) ↑a
      n : Nat
      a✝ : Eq ((HMul.hMul A { coeff := WittVector.inverseCoeff a A }).coeff n) (Witt …
      H_coeff : k := HAdd.hAdd (HMul.hMul (A.coeff (HAdd.hAdd n 1)) ↑(HPow.hPow (Inv …
      H : Eq (HMul.hMul ↑(HPow.hPow a (HPow.hPow p (HAdd.hAdd n 1))) ↑(Inv.inv (HPow …
      ⊢ Eq ((HMul.hMul A { coeff := WittVector.inverseCoeff a A }).coeff (HAdd.hAdd  …
    -/
    linear_combination (norm := skip) -H_coeff * H
    /-
      case a.a
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝¹ : CommRing k
      inst✝ : CharP k p
      a : Units k
      A : WittVector p k
      hA : Eq (A.coeff 0) ↑a
      n : Nat
      a✝ : Eq ((HMul.hMul A { coeff := WittVector.inverseCoeff a A }).coeff n) (Witt …
      H_coeff : k := HAdd.hAdd (HMul.hMul (A.coeff (HAdd.hAdd n 1)) ↑(HPow.hPow (Inv …
      H : Eq (HMul.hMul ↑(HPow.hPow a (HPow.hPow p (HAdd.hAdd n 1))) ↑(Inv.inv (HPow …
      ⊢ Eq (HSub.hSub (HAdd.hAdd ((HMul.hMul A { coeff := WittVector.inverseCoeff a  …
    -/
    have ha : (a : k) ^ p ^ (n + 1) = ↑(a ^ p ^ (n + 1)) := by norm_cast
    /-
      case a.a
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝¹ : CommRing k
      inst✝ : CharP k p
      a : Units k
      A : WittVector p k
      hA : Eq (A.coeff 0) ↑a
      n : Nat
      a✝ : Eq ((HMul.hMul A { coeff := WittVector.inverseCoeff a A }).coeff n) (Witt …
      H_coeff : k := HAdd.hAdd (HMul.hMul (A.coeff (HAdd.hAdd n 1)) ↑(HPow.hPow (Inv …
      H : Eq (HMul.hMul ↑(HPow.hPow a (HPow.hPow p (HAdd.hAdd n 1))) ↑(Inv.inv (HPow …
      ha : Eq (HPow.hPow (↑a) (HPow.hPow p (HAdd.hAdd n 1))) ↑(HPow.hPow a (HPow.hPo …
      ⊢ Eq (HSub.hSub (HAdd.hAdd ((HMul.hMul A { coeff := WittVector.inverseCoeff a  …
    -/
    have ha_inv : (↑a⁻¹ : k) ^ p ^ (n + 1) = ↑(a ^ p ^ (n + 1))⁻¹ := by norm_cast
    simp only [nthRemainder_spec, inverseCoeff, succNthValUnits, hA,
      one_coeff_eq_of_pos, Nat.succ_pos', ha_inv, ha, inv_pow]
    /-
      case a.a
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝¹ : CommRing k
      inst✝ : CharP k p
      a : Units k
      A : WittVector p k
      hA : Eq (A.coeff 0) ↑a
      n : Nat
      a✝ : Eq ((HMul.hMul A { coeff := WittVector.inverseCoeff a A }).coeff n) (Witt …
      H_coeff : k := HAdd.hAdd (HMul.hMul (A.coeff (HAdd.hAdd n 1)) ↑(HPow.hPow (Inv …
      H : Eq (HMul.hMul ↑(HPow.hPow a (HPow.hPow p (HAdd.hAdd n 1))) ↑(Inv.inv (HPow …
      ha : Eq (HPow.hPow (↑a) (HPow.hPow p (HAdd.hAdd n 1))) ↑(HPow.hPow a (HPow.hPo …
      ha_inv : Eq (HPow.hPow (↑(Inv.inv a)) (HPow.hPow p (HAdd.hAdd n 1))) ↑(Inv.inv …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (A.coeff (HAdd.hAd …
    -/
    ring!)
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_mkUnit {a : Units k} {A : 𝕎 k} (hA : A.coeff 0 = a) : (mkUnit hA : 𝕎 k) = A :=
  rfl


theorem isUnit_of_coeff_zero_ne_zero (x : 𝕎 k) (hx : x.coeff 0 ≠ 0) : IsUnit x := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : CharP k p
    x : WittVector p k
    hx : Ne (x.coeff 0) 0
    ⊢ IsUnit x
  -/
  let y : kˣ := Units.mk0 (x.coeff 0) hx
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : CharP k p
    x : WittVector p k
    hx : Ne (x.coeff 0) 0
    y : Units k := Units.mk0 (x.coeff 0) hx
    ⊢ IsUnit x
  -/
  have hy : x.coeff 0 = y := rfl
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : CharP k p
    x : WittVector p k
    hx : Ne (x.coeff 0) 0
    y : Units k := Units.mk0 (x.coeff 0) hx
    hy : Eq (x.coeff 0) ↑y
    ⊢ IsUnit x
  -/
  exact (mkUnit hy).isUnit
  /-
    🎉 no goals
  -/


theorem irreducible : Irreducible (p : 𝕎 k) := by
  have hp : ¬IsUnit (p : 𝕎 k) := by
    intro hp
    simpa only [constantCoeff_apply, coeff_p_zero, not_isUnit_zero] using
      (constantCoeff : WittVector p k →+* _).isUnit_map hp
  /-
    p : Nat
    hp✝ : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : CharP k p
    hp : Not (IsUnit ↑p)
    ⊢ Irreducible ↑p
  -/
  refine ⟨hp, fun a b hab => ?_⟩
  obtain ⟨ha0, hb0⟩ : a ≠ 0 ∧ b ≠ 0 := by
    rw [← mul_ne_zero_iff]; intro h; rw [h] at hab; exact p_nonzero p k hab
  /-
    case intro
    p : Nat
    hp✝ : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : CharP k p
    hp : Not (IsUnit ↑p)
    a b : WittVector p k
    hab : Eq (↑p) (HMul.hMul a b)
    ha0 : Ne a 0
    hb0 : Ne b 0
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  obtain ⟨m, a, ha, rfl⟩ := verschiebung_nonzero ha0
  /-
    case intro.intro.intro.intro
    p : Nat
    hp✝ : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : CharP k p
    hp : Not (IsUnit ↑p)
    b : WittVector p k
    hb0 : Ne b 0
    m : Nat
    a : WittVector p k
    ha : Ne (a.coeff 0) 0
    hab : Eq (↑p) (HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) m a) b)
    ha0 : Ne (Nat.iterate (⇑WittVector.verschiebung) m a) 0
    ⊢ Or (IsUnit (Nat.iterate (⇑WittVector.verschiebung) m a)) (IsUnit b)
  -/
  obtain ⟨n, b, hb, rfl⟩ := verschiebung_nonzero hb0
  /-
    case intro.intro.intro.intro.intro.intro.intro
    p : Nat
    hp✝ : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : CharP k p
    hp : Not (IsUnit ↑p)
    m : Nat
    a : WittVector p k
    ha : Ne (a.coeff 0) 0
    ha0 : Ne (Nat.iterate (⇑WittVector.verschiebung) m a) 0
    n : Nat
    b : WittVector p k
    hb : Ne (b.coeff 0) 0
    hb0 : Ne (Nat.iterate (⇑WittVector.verschiebung) n b) 0
    hab : Eq (↑p) (HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) m a) (Nat.ite …
    ⊢ Or (IsUnit (Nat.iterate (⇑WittVector.verschiebung) m a)) (IsUnit (Nat.iterat …
  -/
  cases m; · exact Or.inl (isUnit_of_coeff_zero_ne_zero a ha)
             /-
               🎉 no goals
             -/
  /-
    case intro.intro.intro.intro.intro.intro.intro.succ
    p : Nat
    hp✝ : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : CharP k p
    hp : Not (IsUnit ↑p)
    a : WittVector p k
    ha : Ne (a.coeff 0) 0
    n : Nat
    b : WittVector p k
    hb : Ne (b.coeff 0) 0
    hb0 : Ne (Nat.iterate (⇑WittVector.verschiebung) n b) 0
    n✝ : Nat
    ha0 : Ne (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n✝ 1) a) 0
    hab : Eq (↑p) (HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n✝ …
    ⊢ Or (IsUnit (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n✝ 1) a)) (IsU …
  -/
  cases' n with n; · exact Or.inr (isUnit_of_coeff_zero_ne_zero b hb)
                     /-
                       🎉 no goals
                     -/
  /-
    case intro.intro.intro.intro.intro.intro.intro.succ.succ
    p : Nat
    hp✝ : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : CharP k p
    hp : Not (IsUnit ↑p)
    a : WittVector p k
    ha : Ne (a.coeff 0) 0
    b : WittVector p k
    hb : Ne (b.coeff 0) 0
    n✝ : Nat
    ha0 : Ne (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n✝ 1) a) 0
    n : Nat
    hb0 : Ne (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n 1) b) 0
    hab : Eq (↑p) (HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n✝ …
    ⊢ Or (IsUnit (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n✝ 1) a)) (IsU …
  -/
  rw [iterate_verschiebung_mul] at hab
  /-
    case intro.intro.intro.intro.intro.intro.intro.succ.succ
    p : Nat
    hp✝ : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : CharP k p
    hp : Not (IsUnit ↑p)
    a : WittVector p k
    ha : Ne (a.coeff 0) 0
    b : WittVector p k
    hb : Ne (b.coeff 0) 0
    n✝ : Nat
    ha0 : Ne (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n✝ 1) a) 0
    n : Nat
    hb0 : Ne (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n 1) b) 0
    hab : Eq (↑p) (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd (HAdd.hAdd n✝ …
    ⊢ Or (IsUnit (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n✝ 1) a)) (IsU …
  -/
  apply_fun fun x => coeff x 1 at hab
  simp only [coeff_p_one, Nat.add_succ, add_comm _ n, Function.iterate_succ', Function.comp_apply,
    verschiebung_coeff_add_one, verschiebung_coeff_zero] at hab
  /-
    case intro.intro.intro.intro.intro.intro.intro.succ.succ
    p : Nat
    hp✝ : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : CharP k p
    hp : Not (IsUnit ↑p)
    a : WittVector p k
    ha : Ne (a.coeff 0) 0
    b : WittVector p k
    hb : Ne (b.coeff 0) 0
    n✝ : Nat
    ha0 : Ne (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n✝ 1) a) 0
    n : Nat
    hb0 : Ne (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n 1) b) 0
    hab : Eq 1 0
    ⊢ Or (IsUnit (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd n✝ 1) a)) (IsU …
  -/
  exact (one_ne_zero hab).elim
  /-
    🎉 no goals
  -/


theorem exists_eq_pow_p_mul (a : 𝕎 k) (ha : a ≠ 0) :
    ∃ (m : ℕ) (b : 𝕎 k), b.coeff 0 ≠ 0 ∧ a = (p : 𝕎 k) ^ m * b := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : CommRing k
    inst✝¹ : CharP k p
    inst✝ : PerfectRing k p
    a : WittVector p k
    ha : Ne a 0
    ⊢ Exists fun m => Exists fun b => And (Ne (b.coeff 0) 0) (Eq a (HMul.hMul (HPo …
  -/
  obtain ⟨m, c, hc, hcm⟩ := WittVector.verschiebung_nonzero ha
  /-
    case intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : CommRing k
    inst✝¹ : CharP k p
    inst✝ : PerfectRing k p
    a : WittVector p k
    ha : Ne a 0
    m : Nat
    c : WittVector p k
    hc : Ne (c.coeff 0) 0
    hcm : Eq a (Nat.iterate (⇑WittVector.verschiebung) m c)
    ⊢ Exists fun m => Exists fun b => And (Ne (b.coeff 0) 0) (Eq a (HMul.hMul (HPo …
  -/
  obtain ⟨b, rfl⟩ := (frobenius_bijective p k).surjective.iterate m c
  /-
    case intro.intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : CommRing k
    inst✝¹ : CharP k p
    inst✝ : PerfectRing k p
    a : WittVector p k
    ha : Ne a 0
    m : Nat
    b : WittVector p k
    hc : Ne ((Nat.iterate (⇑WittVector.frobenius) m b).coeff 0) 0
    hcm : Eq a (Nat.iterate (⇑WittVector.verschiebung) m (Nat.iterate (⇑WittVector …
    ⊢ Exists fun m => Exists fun b => And (Ne (b.coeff 0) 0) (Eq a (HMul.hMul (HPo …
  -/
  rw [WittVector.iterate_frobenius_coeff] at hc
  /-
    case intro.intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : CommRing k
    inst✝¹ : CharP k p
    inst✝ : PerfectRing k p
    a : WittVector p k
    ha : Ne a 0
    m : Nat
    b : WittVector p k
    hc : Ne (HPow.hPow (b.coeff 0) (HPow.hPow p m)) 0
    hcm : Eq a (Nat.iterate (⇑WittVector.verschiebung) m (Nat.iterate (⇑WittVector …
    ⊢ Exists fun m => Exists fun b => And (Ne (b.coeff 0) 0) (Eq a (HMul.hMul (HPo …
  -/
  have := congr_fun (WittVector.verschiebung_frobenius_comm.comp_iterate m) b
  /-
    case intro.intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : CommRing k
    inst✝¹ : CharP k p
    inst✝ : PerfectRing k p
    a : WittVector p k
    ha : Ne a 0
    m : Nat
    b : WittVector p k
    hc : Ne (HPow.hPow (b.coeff 0) (HPow.hPow p m)) 0
    hcm : Eq a (Nat.iterate (⇑WittVector.verschiebung) m (Nat.iterate (⇑WittVector …
    this : Eq (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fro …
    ⊢ Exists fun m => Exists fun b => And (Ne (b.coeff 0) 0) (Eq a (HMul.hMul (HPo …
  -/
  simp only [Function.comp_apply] at this
  /-
    case intro.intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : CommRing k
    inst✝¹ : CharP k p
    inst✝ : PerfectRing k p
    a : WittVector p k
    ha : Ne a 0
    m : Nat
    b : WittVector p k
    hc : Ne (HPow.hPow (b.coeff 0) (HPow.hPow p m)) 0
    hcm : Eq a (Nat.iterate (⇑WittVector.verschiebung) m (Nat.iterate (⇑WittVector …
    this : Eq (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fro …
    ⊢ Exists fun m => Exists fun b => And (Ne (b.coeff 0) 0) (Eq a (HMul.hMul (HPo …
  -/
  rw [← this] at hcm
  /-
    case intro.intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : CommRing k
    inst✝¹ : CharP k p
    inst✝ : PerfectRing k p
    a : WittVector p k
    ha : Ne a 0
    m : Nat
    b : WittVector p k
    hc : Ne (HPow.hPow (b.coeff 0) (HPow.hPow p m)) 0
    hcm : Eq a (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fr …
    this : Eq (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fro …
    ⊢ Exists fun m => Exists fun b => And (Ne (b.coeff 0) 0) (Eq a (HMul.hMul (HPo …
  -/
  refine ⟨m, b, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : CommRing k
      inst✝¹ : CharP k p
      inst✝ : PerfectRing k p
      a : WittVector p k
      ha : Ne a 0
      m : Nat
      b : WittVector p k
      hc : Ne (HPow.hPow (b.coeff 0) (HPow.hPow p m)) 0
      hcm : Eq a (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fr …
      this : Eq (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fro …
      ⊢ Ne (b.coeff 0) 0
    -/
  · contrapose! hc
    /-
      case intro.intro.intro.intro.refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : CommRing k
      inst✝¹ : CharP k p
      inst✝ : PerfectRing k p
      a : WittVector p k
      ha : Ne a 0
      m : Nat
      b : WittVector p k
      hcm : Eq a (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fr …
      this : Eq (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fro …
      hc : Eq (b.coeff 0) 0
      ⊢ Eq (HPow.hPow (b.coeff 0) (HPow.hPow p m)) 0
    -/
    simp [hc, zero_pow <| pow_ne_zero _ hp.out.ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : CommRing k
      inst✝¹ : CharP k p
      inst✝ : PerfectRing k p
      a : WittVector p k
      ha : Ne a 0
      m : Nat
      b : WittVector p k
      hc : Ne (HPow.hPow (b.coeff 0) (HPow.hPow p m)) 0
      hcm : Eq a (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fr …
      this : Eq (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fro …
      ⊢ Eq a (HMul.hMul (HPow.hPow (↑p) m) b)
    -/
  · simp_rw [← mul_left_iterate (p : 𝕎 k) m]
    /-
      case intro.intro.intro.intro.refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : CommRing k
      inst✝¹ : CharP k p
      inst✝ : PerfectRing k p
      a : WittVector p k
      ha : Ne a 0
      m : Nat
      b : WittVector p k
      hc : Ne (HPow.hPow (b.coeff 0) (HPow.hPow p m)) 0
      hcm : Eq a (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fr …
      this : Eq (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fro …
      ⊢ Eq a (Nat.iterate (fun x => HMul.hMul (↑p) x) m b)
    -/
    convert hcm using 2
    /-
      case h.e'_3.h.e'_2
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : CommRing k
      inst✝¹ : CharP k p
      inst✝ : PerfectRing k p
      a : WittVector p k
      ha : Ne a 0
      m : Nat
      b : WittVector p k
      hc : Ne (HPow.hPow (b.coeff 0) (HPow.hPow p m)) 0
      hcm : Eq a (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fr …
      this : Eq (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fro …
      ⊢ Eq (fun x => HMul.hMul (↑p) x) (Function.comp ⇑WittVector.verschiebung ⇑Witt …
    -/
    ext1 x
    /-
      case h.e'_3.h.e'_2.h
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : CommRing k
      inst✝¹ : CharP k p
      inst✝ : PerfectRing k p
      a : WittVector p k
      ha : Ne a 0
      m : Nat
      b : WittVector p k
      hc : Ne (HPow.hPow (b.coeff 0) (HPow.hPow p m)) 0
      hcm : Eq a (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fr …
      this : Eq (Nat.iterate (Function.comp ⇑WittVector.verschiebung ⇑WittVector.fro …
      x : WittVector p k
      ⊢ Eq (HMul.hMul (↑p) x) (Function.comp (⇑WittVector.verschiebung) (⇑WittVector …
    -/
    rw [mul_comm, ← WittVector.verschiebung_frobenius x]; rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem exists_eq_pow_p_mul' (a : 𝕎 k) (ha : a ≠ 0) :
    ∃ (m : ℕ) (b : Units (𝕎 k)), a = (p : 𝕎 k) ^ m * b := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : PerfectRing k p
    a : WittVector p k
    ha : Ne a 0
    ⊢ Exists fun m => Exists fun b => Eq a (HMul.hMul (HPow.hPow (↑p) m) ↑b)
  -/
  obtain ⟨m, b, h₁, h₂⟩ := exists_eq_pow_p_mul a ha
  /-
    case intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : PerfectRing k p
    a : WittVector p k
    ha : Ne a 0
    m : Nat
    b : WittVector p k
    h₁ : Ne (b.coeff 0) 0
    h₂ : Eq a (HMul.hMul (HPow.hPow (↑p) m) b)
    ⊢ Exists fun m => Exists fun b => Eq a (HMul.hMul (HPow.hPow (↑p) m) ↑b)
  -/
  let b₀ := Units.mk0 (b.coeff 0) h₁
  /-
    case intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : PerfectRing k p
    a : WittVector p k
    ha : Ne a 0
    m : Nat
    b : WittVector p k
    h₁ : Ne (b.coeff 0) 0
    h₂ : Eq a (HMul.hMul (HPow.hPow (↑p) m) b)
    b₀ : Units k := Units.mk0 (b.coeff 0) h₁
    ⊢ Exists fun m => Exists fun b => Eq a (HMul.hMul (HPow.hPow (↑p) m) ↑b)
  -/
  have hb₀ : b.coeff 0 = b₀ := rfl
  /-
    case intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : PerfectRing k p
    a : WittVector p k
    ha : Ne a 0
    m : Nat
    b : WittVector p k
    h₁ : Ne (b.coeff 0) 0
    h₂ : Eq a (HMul.hMul (HPow.hPow (↑p) m) b)
    b₀ : Units k := Units.mk0 (b.coeff 0) h₁
    hb₀ : Eq (b.coeff 0) ↑b₀
    ⊢ Exists fun m => Exists fun b => Eq a (HMul.hMul (HPow.hPow (↑p) m) ↑b)
  -/
  exact ⟨m, mkUnit hb₀, h₂⟩
  /-
    🎉 no goals
  -/

/-
Note: The following lemma should be an instance, but it seems to cause some
exponential blowups in certain typeclass resolution problems.
See the following Lean4 issue as well as the zulip discussion linked there:
https://github.com/leanprover/lean4/issues/1102
-/

/-- The ring of Witt Vectors of a perfect field of positive characteristic is a DVR.
-/
theorem isDiscreteValuationRing : IsDiscreteValuationRing (𝕎 k) :=
  IsDiscreteValuationRing.ofHasUnitMulPowIrreducibleFactorization (by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : PerfectRing k p
      ⊢ IsDiscreteValuationRing.HasUnitMulPowIrreducibleFactorization (WittVector p k)
    -/
    refine ⟨p, irreducible p, fun {x} hx => ?_⟩
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : PerfectRing k p
      x : WittVector p k
      hx : Ne x 0
      ⊢ Exists fun n => Associated (HPow.hPow (↑p) n) x
    -/
    obtain ⟨n, b, hb⟩ := exists_eq_pow_p_mul' x hx
    /-
      case intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : PerfectRing k p
      x : WittVector p k
      hx : Ne x 0
      n : Nat
      b : Units (WittVector p k)
      hb : Eq x (HMul.hMul (HPow.hPow (↑p) n) ↑b)
      ⊢ Exists fun n => Associated (HPow.hPow (↑p) n) x
    -/
    exact ⟨n, b, hb.symm⟩)
    /-
      🎉 no goals
    -/


