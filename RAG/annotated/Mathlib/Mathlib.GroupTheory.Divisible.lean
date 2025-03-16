/--
An `AddMonoid A` is `α`-divisible iff `n • x = a` has a solution for all `n ≠ 0 ∈ α` and `a ∈ A`.
Here we adopt a constructive approach where we ask an explicit `div : A → α → A` function such that
* `div a 0 = 0` for all `a ∈ A`
* `n • div a n = a` for all `n ≠ 0 ∈ α` and `a ∈ A`.
-/
class DivisibleBy where
  div : A → α → A
  div_zero : ∀ a, div a 0 = 0
  div_cancel : ∀ {n : α} (a : A), n ≠ 0 → n • div a n = a


/-- A `Monoid A` is `α`-rootable iff `xⁿ = a` has a solution for all `n ≠ 0 ∈ α` and `a ∈ A`.
Here we adopt a constructive approach where we ask an explicit `root : A → α → A` function such that
* `root a 0 = 1` for all `a ∈ A`
* `(root a n)ⁿ = a` for all `n ≠ 0 ∈ α` and `a ∈ A`.
-/
@[to_additive]
class RootableBy where
  root : A → α → A
  root_zero : ∀ a, root a 0 = 1
  root_cancel : ∀ {n : α} (a : A), n ≠ 0 → root a n ^ n = a


@[to_additive smul_right_surj_of_divisibleBy]
theorem pow_left_surj_of_rootableBy [RootableBy A α] {n : α} (hn : n ≠ 0) :
    Function.Surjective (fun a => a ^ n : A → A) := fun x =>
  ⟨RootableBy.root x n, RootableBy.root_cancel _ hn⟩


/--
A `Monoid A` is `α`-rootable iff the `pow _ n` function is surjective, i.e. the constructive version
implies the textbook approach.
-/
@[to_additive divisibleByOfSMulRightSurj
      "An `AddMonoid A` is `α`-divisible iff `n • _` is a surjective function, i.e. the constructive
      version implies the textbook approach."]
noncomputable def rootableByOfPowLeftSurj
    (H : ∀ {n : α}, n ≠ 0 → Function.Surjective (fun a => a ^ n : A → A)) : RootableBy A α where
  root a n := @dite _ (n = 0) (Classical.dec _) (fun _ => (1 : A)) fun hn => (H hn a).choose
                    /-
                      A : Type u_1
                      α : Type u_2
                      inst✝² : Monoid A
                      inst✝¹ : Pow A α
                      inst✝ : Zero α
                      H : ∀ {n : α}, Ne n 0 → Function.Surjective fun a => HPow.hPow a n
                      x✝ : A
                      ⊢ Eq ((fun a n => dite (Eq n 0) (fun x => 1) fun hn => ⋯.choose) x✝ 0) 1
                    -/
  root_zero _ := by classical exact dif_pos rfl
                    /-
                      🎉 no goals
                    -/
  root_cancel a hn := by
    /-
      A : Type u_1
      α : Type u_2
      inst✝² : Monoid A
      inst✝¹ : Pow A α
      inst✝ : Zero α
      H : ∀ {n : α}, Ne n 0 → Function.Surjective fun a => HPow.hPow a n
      n✝ : α
      a : A
      hn : Ne n✝ 0
      ⊢ Eq (HPow.hPow ((fun a n => dite (Eq n 0) (fun x => 1) fun hn => ⋯.choose) a  …
    -/
    dsimp only
    /-
      A : Type u_1
      α : Type u_2
      inst✝² : Monoid A
      inst✝¹ : Pow A α
      inst✝ : Zero α
      H : ∀ {n : α}, Ne n 0 → Function.Surjective fun a => HPow.hPow a n
      n✝ : α
      a : A
      hn : Ne n✝ 0
      ⊢ Eq (HPow.hPow (dite (Eq n✝ 0) (fun x => 1) fun hn => ⋯.choose) n✝) a
    -/
    rw [dif_neg hn]
    /-
      A : Type u_1
      α : Type u_2
      inst✝² : Monoid A
      inst✝¹ : Pow A α
      inst✝ : Zero α
      H : ∀ {n : α}, Ne n 0 → Function.Surjective fun a => HPow.hPow a n
      n✝ : α
      a : A
      hn : Ne n✝ 0
      ⊢ Eq (HPow.hPow ⋯.choose n✝) a
    -/
    exact (H hn a).choose_spec
    /-
      🎉 no goals
    -/


@[to_additive]
instance Pi.rootableBy : RootableBy (∀ i, B i) β where
  root x n i := RootableBy.root (x i) n
  root_zero _x := funext fun _i => RootableBy.root_zero _
  root_cancel _x hn := funext fun _i => RootableBy.root_cancel _ hn


@[to_additive]
instance Prod.rootableBy : RootableBy (B × B') β where
  root p n := (RootableBy.root p.1 n, RootableBy.root p.2 n)
  root_zero _p := Prod.ext (RootableBy.root_zero _) (RootableBy.root_zero _)
  root_cancel _p hn := Prod.ext (RootableBy.root_cancel _ hn) (RootableBy.root_cancel _ hn)


@[to_additive]
instance ULift.instRootableBy [RootableBy A α] : RootableBy (ULift A) α where
  root x a := ULift.up <| RootableBy.root x.down a
  root_zero x := ULift.ext _ _ <| RootableBy.root_zero x.down
  root_cancel _ h := ULift.ext _ _ <| RootableBy.root_cancel _ h


theorem smul_top_eq_top_of_divisibleBy_int [DivisibleBy A ℤ] {n : ℤ} (hn : n ≠ 0) :
    n • (⊤ : AddSubgroup A) = ⊤ :=
  AddSubgroup.map_top_of_surjective _ fun a => ⟨DivisibleBy.div a n, DivisibleBy.div_cancel _ hn⟩


/-- If for all `n ≠ 0 ∈ ℤ`, `n • A = A`, then `A` is divisible.
-/
noncomputable def divisibleByIntOfSMulTopEqTop
    (H : ∀ {n : ℤ} (_hn : n ≠ 0), n • (⊤ : AddSubgroup A) = ⊤) : DivisibleBy A ℤ where
  div a n :=
                                                                   /-
                                                                     A : Type u_1
                                                                     inst✝ : AddCommGroup A
                                                                     H : ∀ {n : Int}, Ne n 0 → Eq (HSMul.hSMul n Top.top) Top.top
                                                                     a : A
                                                                     n : Int
                                                                     hn : Not (Eq n 0)
                                                                     ⊢ Membership.mem (HSMul.hSMul n Top.top) a
                                                                   -/
    if hn : n = 0 then 0 else (show a ∈ n • (⊤ : AddSubgroup A) by rw [H hn]; trivial).choose
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
  div_zero _ := dif_pos rfl
  div_cancel a hn := by
    /-
      A : Type u_1
      inst✝ : AddCommGroup A
      H : ∀ {n : Int}, Ne n 0 → Eq (HSMul.hSMul n Top.top) Top.top
      n✝ : Int
      a : A
      hn : Ne n✝ 0
      ⊢ Eq (HSMul.hSMul n✝ ((fun a n => dite (Eq n 0) (fun hn => 0) fun hn => Exists …
    -/
    simp_rw [dif_neg hn]
    /-
      A : Type u_1
      inst✝ : AddCommGroup A
      H : ∀ {n : Int}, Ne n 0 → Eq (HSMul.hSMul n Top.top) Top.top
      n✝ : Int
      a : A
      hn : Ne n✝ 0
      ⊢ Eq (HSMul.hSMul n✝ (Exists.choose ⋯)) a
    -/
    generalize_proofs h1
    /-
      A : Type u_1
      inst✝ : AddCommGroup A
      H : ∀ {n : Int}, Ne n 0 → Eq (HSMul.hSMul n Top.top) Top.top
      n✝ : Int
      a : A
      hn : Ne n✝ 0
      h1 : Exists fun a_1 => And (Membership.mem (↑Top.top) a_1) (Eq (((DistribMulAc …
      ⊢ Eq (HSMul.hSMul n✝ h1.choose) a
    -/
    exact h1.choose_spec.2
    /-
      🎉 no goals
    -/


instance (priority := 100) divisibleByIntOfCharZero {𝕜} [DivisionRing 𝕜] [CharZero 𝕜] :
    DivisibleBy 𝕜 ℤ where
  div q n := q / n
                   /-
                     𝕜 : Type ?u.9678
                     inst✝¹ : DivisionRing 𝕜
                     inst✝ : CharZero 𝕜
                     q : 𝕜
                     ⊢ Eq ((fun q n => HDiv.hDiv q ↑n) q 0) 0
                   -/
  div_zero q := by norm_num
                   /-
                     🎉 no goals
                   -/
  div_cancel {n} q hn := by
    /-
      𝕜 : Type ?u.9678
      inst✝¹ : DivisionRing 𝕜
      inst✝ : CharZero 𝕜
      n : Int
      q : 𝕜
      hn : Ne n 0
      ⊢ Eq (HSMul.hSMul n ((fun q n => HDiv.hDiv q ↑n) q n)) q
    -/
    rw [zsmul_eq_mul, (Int.cast_commute n _).eq, div_mul_cancel₀ q (Int.cast_ne_zero.mpr hn)]
    /-
      🎉 no goals
    -/


open Int in
/-- A group is `ℤ`-rootable if it is `ℕ`-rootable.
-/
@[to_additive "An additive group is `ℤ`-divisible if it is `ℕ`-divisible."]
def rootableByIntOfRootableByNat [RootableBy A ℕ] : RootableBy A ℤ where
  root a z :=
    match z with
    | (n : ℕ) => RootableBy.root a n
    | -[n+1] => (RootableBy.root a (n + 1))⁻¹
  root_zero a := RootableBy.root_zero a
  root_cancel {n} a hn := by
    /-
      A : Type u_1
      inst✝¹ : Group A
      inst✝ : RootableBy A Nat
      n : Int
      a : A
      hn : Ne n 0
      ⊢ Eq (HPow.hPow ((fun a z => Group.rootableByIntOfRootableByNat.match_1 (fun z …
    -/
    induction n
      /-
        case ofNat
        A : Type u_1
        inst✝¹ : Group A
        inst✝ : RootableBy A Nat
        a : A
        a✝ : Nat
        hn : Ne (Int.ofNat a✝) 0
        ⊢ Eq (HPow.hPow ((fun a z => Group.rootableByIntOfRootableByNat.match_1 (fun z …
      -/
    · change RootableBy.root a _ ^ _ = a
      /-
        case ofNat
        A : Type u_1
        inst✝¹ : Group A
        inst✝ : RootableBy A Nat
        a : A
        a✝ : Nat
        hn : Ne (Int.ofNat a✝) 0
        ⊢ Eq (HPow.hPow (RootableBy.root a a✝) (Int.ofNat a✝)) a
      -/
      norm_num
      /-
        case ofNat
        A : Type u_1
        inst✝¹ : Group A
        inst✝ : RootableBy A Nat
        a : A
        a✝ : Nat
        hn : Ne (Int.ofNat a✝) 0
        ⊢ Eq (HPow.hPow (RootableBy.root a a✝) a✝) a
      -/
      rw [RootableBy.root_cancel]
      /-
        case ofNat.a
        A : Type u_1
        inst✝¹ : Group A
        inst✝ : RootableBy A Nat
        a : A
        a✝ : Nat
        hn : Ne (Int.ofNat a✝) 0
        ⊢ Ne a✝ 0
      -/
      rw [Int.ofNat_eq_coe] at hn
      /-
        case ofNat.a
        A : Type u_1
        inst✝¹ : Group A
        inst✝ : RootableBy A Nat
        a : A
        a✝ : Nat
        hn : Ne (↑a✝) 0
        ⊢ Ne a✝ 0
      -/
      exact mod_cast hn
      /-
        🎉 no goals
      -/
      /-
        case negSucc
        A : Type u_1
        inst✝¹ : Group A
        inst✝ : RootableBy A Nat
        a : A
        a✝ : Nat
        hn : Ne (Int.negSucc a✝) 0
        ⊢ Eq (HPow.hPow ((fun a z => Group.rootableByIntOfRootableByNat.match_1 (fun z …
      -/
    · change (RootableBy.root a _)⁻¹ ^ _ = a
      /-
        case negSucc
        A : Type u_1
        inst✝¹ : Group A
        inst✝ : RootableBy A Nat
        a : A
        a✝ : Nat
        hn : Ne (Int.negSucc a✝) 0
        ⊢ Eq (HPow.hPow (Inv.inv (RootableBy.root a (HAdd.hAdd a✝ 1))) (Int.negSucc a✝ …
      -/
      norm_num
      /-
        case negSucc
        A : Type u_1
        inst✝¹ : Group A
        inst✝ : RootableBy A Nat
        a : A
        a✝ : Nat
        hn : Ne (Int.negSucc a✝) 0
        ⊢ Eq (HPow.hPow (RootableBy.root a (HAdd.hAdd a✝ 1)) (HAdd.hAdd a✝ 1)) a
      -/
      rw [RootableBy.root_cancel]
      /-
        case negSucc.a
        A : Type u_1
        inst✝¹ : Group A
        inst✝ : RootableBy A Nat
        a : A
        a✝ : Nat
        hn : Ne (Int.negSucc a✝) 0
        ⊢ Ne (HAdd.hAdd a✝ 1) 0
      -/
      norm_num
      /-
        🎉 no goals
      -/


/-- A group is `ℕ`-rootable if it is `ℤ`-rootable
-/
@[to_additive "An additive group is `ℕ`-divisible if it `ℤ`-divisible."]
def rootableByNatOfRootableByInt [RootableBy A ℤ] : RootableBy A ℕ where
  root a n := RootableBy.root a (n : ℤ)
  root_zero a := RootableBy.root_zero a
  root_cancel {n} a hn := by
    -- Porting note: replaced `norm_num`
    /-
      A : Type u_1
      inst✝¹ : Group A
      inst✝ : RootableBy A Int
      n : Nat
      a : A
      hn : Ne n 0
      ⊢ Eq (HPow.hPow ((fun a n => RootableBy.root a ↑n) a n) n) a
    -/
    simpa only [zpow_natCast] using RootableBy.root_cancel a (show (n : ℤ) ≠ 0 from mod_cast hn)
    /-
      🎉 no goals
    -/


/--
If `f : A → B` is a surjective homomorphism and `A` is `α`-rootable, then `B` is also `α`-rootable.
-/
@[to_additive
      "If `f : A → B` is a surjective homomorphism and `A` is `α`-divisible, then `B` is also
      `α`-divisible."]
noncomputable def Function.Surjective.rootableBy (hf : Function.Surjective f)
    (hpow : ∀ (a : A) (n : α), f (a ^ n) = f a ^ n) : RootableBy B α :=
  rootableByOfPowLeftSurj _ _ fun {n} hn x =>
    let ⟨y, hy⟩ := hf x
    ⟨f <| RootableBy.root y n,
          /-
            A : Type u_1
            B : Type u_2
            α : Type u_3
            inst✝⁵ : Zero α
            inst✝⁴ : Monoid A
            inst✝³ : Monoid B
            inst✝² : Pow A α
            inst✝¹ : Pow B α
            inst✝ : RootableBy A α
            f : A → B
            hf : Function.Surjective f
            hpow : ∀ (a : A) (n : α), Eq (f (HPow.hPow a n)) (HPow.hPow (f a) n)
            n : α
            hn : Ne n 0
            x : B
            y : A
            hy : Eq (f y) x
            ⊢ Eq (HPow.hPow (f (RootableBy.root y n)) n) x
          -/
      (by rw [← hpow (RootableBy.root y n) n, RootableBy.root_cancel _ hn, hy] : _ ^ n = x)⟩
          /-
            🎉 no goals
          -/


@[to_additive DivisibleBy.surjective_smul]
theorem RootableBy.surjective_pow (A α : Type*) [Monoid A] [Pow A α] [Zero α] [RootableBy A α]
    {n : α} (hn : n ≠ 0) : Function.Surjective fun a : A => a ^ n := fun a =>
  ⟨RootableBy.root a n, RootableBy.root_cancel a hn⟩


/-- Any quotient group of a rootable group is rootable. -/
@[to_additive "Any quotient group of a divisible group is divisible"]
noncomputable instance QuotientGroup.rootableBy [RootableBy A ℕ] : RootableBy (A ⧸ B) ℕ :=
  QuotientGroup.mk_surjective.rootableBy _ fun _ _ => rfl


