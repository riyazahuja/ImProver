@[to_additive]
theorem oreDiv_one_surjective_of_finite_left [Finite S] :
    Surjective (fun x ↦ x /ₒ (1 : ↥S) : X → OreLocalization S X) := by
  /-
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite (Subtype fun x => Membership.mem S x)
    ⊢ Function.Surjective fun x => OreLocalization.oreDiv x 1
  -/
  refine OreLocalization.ind fun x s ↦ ?_
  /-
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite (Subtype fun x => Membership.mem S x)
    x : X
    s : Subtype fun x => Membership.mem S x
    ⊢ Exists fun a => Eq ((fun x => OreLocalization.oreDiv x 1) a) (OreLocalizatio …
  -/
  obtain ⟨i, j, hne, heq⟩ := Finite.exists_ne_map_eq_of_infinite (α := ℕ) (s ^ ·)
  /-
    case intro.intro.intro
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite (Subtype fun x => Membership.mem S x)
    x : X
    s : Subtype fun x => Membership.mem S x
    i j : Nat
    hne : Ne i j
    heq : Eq (HPow.hPow s i) (HPow.hPow s j)
    ⊢ Exists fun a => Eq ((fun x => OreLocalization.oreDiv x 1) a) (OreLocalizatio …
  -/
  wlog hlt : j < i generalizing i j
    /-
      case intro.intro.intro.inr
      R : Type u
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type v
      inst✝¹ : MulAction R X
      inst✝ : Finite (Subtype fun x => Membership.mem S x)
      x : X
      s : Subtype fun x => Membership.mem S x
      i j : Nat
      hne : Ne i j
      heq : Eq (HPow.hPow s i) (HPow.hPow s j)
      this : ∀ (i j : Nat), Ne i j → Eq (HPow.hPow s i) (HPow.hPow s j) → LT.lt j i  …
      hlt : Not (LT.lt j i)
      ⊢ Exists fun a => Eq ((fun x => OreLocalization.oreDiv x 1) a) (OreLocalizatio …
    -/
  · exact this j i hne.symm heq.symm (hne.lt_of_le (not_lt.1 hlt))
    /-
      🎉 no goals
    -/
  /-
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite (Subtype fun x => Membership.mem S x)
    x : X
    s : Subtype fun x => Membership.mem S x
    i j : Nat
    hne : Ne i j
    heq : Eq (HPow.hPow s i) (HPow.hPow s j)
    hlt : LT.lt j i
    ⊢ Exists fun a => Eq ((fun x => OreLocalization.oreDiv x 1) a) (OreLocalizatio …
  -/
  use s ^ (i - (j + 1)) • x
  /-
    case h
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite (Subtype fun x => Membership.mem S x)
    x : X
    s : Subtype fun x => Membership.mem S x
    i j : Nat
    hne : Ne i j
    heq : Eq (HPow.hPow s i) (HPow.hPow s j)
    hlt : LT.lt j i
    ⊢ Eq ((fun x => OreLocalization.oreDiv x 1) (HSMul.hSMul (HPow.hPow s (HSub.hS …
  -/
  rw [oreDiv_eq_iff]
  /-
    case h
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite (Subtype fun x => Membership.mem S x)
    x : X
    s : Subtype fun x => Membership.mem S x
    i j : Nat
    hne : Ne i j
    heq : Eq (HPow.hPow s i) (HPow.hPow s j)
    hlt : LT.lt j i
    ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u x) (HSMul.hSMul v (HS …
  -/
  refine ⟨s ^ j, (s ^ (j + 1)).1, ?_, ?_⟩
    /-
      case h.refine_1
      R : Type u
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type v
      inst✝¹ : MulAction R X
      inst✝ : Finite (Subtype fun x => Membership.mem S x)
      x : X
      s : Subtype fun x => Membership.mem S x
      i j : Nat
      hne : Ne i j
      heq : Eq (HPow.hPow s i) (HPow.hPow s j)
      hlt : LT.lt j i
      ⊢ Eq (HSMul.hSMul (HPow.hPow s j) x) (HSMul.hSMul (↑(HPow.hPow s (HAdd.hAdd j  …
    -/
  · change s ^ j • x = s ^ (j + 1) • s ^ (i - (j + 1)) • x
    /-
      case h.refine_1
      R : Type u
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type v
      inst✝¹ : MulAction R X
      inst✝ : Finite (Subtype fun x => Membership.mem S x)
      x : X
      s : Subtype fun x => Membership.mem S x
      i j : Nat
      hne : Ne i j
      heq : Eq (HPow.hPow s i) (HPow.hPow s j)
      hlt : LT.lt j i
      ⊢ Eq (HSMul.hSMul (HPow.hPow s j) x) (HSMul.hSMul (HPow.hPow s (HAdd.hAdd j 1) …
    -/
    rw [← mul_smul, ← pow_add, Nat.add_sub_cancel' hlt, heq]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      R : Type u
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type v
      inst✝¹ : MulAction R X
      inst✝ : Finite (Subtype fun x => Membership.mem S x)
      x : X
      s : Subtype fun x => Membership.mem S x
      i j : Nat
      hne : Ne i j
      heq : Eq (HPow.hPow s i) (HPow.hPow s j)
      hlt : LT.lt j i
      ⊢ Eq (HMul.hMul ↑(HPow.hPow s j) ↑s) (HMul.hMul ↑(HPow.hPow s (HAdd.hAdd j 1)) …
    -/
  · simp_rw [SubmonoidClass.coe_pow, OneMemClass.coe_one, mul_one, pow_succ]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem oreDiv_one_surjective_of_finite_right [Finite X] :
    Surjective (fun x ↦ x /ₒ (1 : ↥S) : X → OreLocalization S X) := by
  /-
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite X
    ⊢ Function.Surjective fun x => OreLocalization.oreDiv x 1
  -/
  refine OreLocalization.ind fun x s ↦ ?_
  /-
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite X
    x : X
    s : Subtype fun x => Membership.mem S x
    ⊢ Exists fun a => Eq ((fun x => OreLocalization.oreDiv x 1) a) (OreLocalizatio …
  -/
  obtain ⟨i, j, hne, heq⟩ := Finite.exists_ne_map_eq_of_infinite (α := ℕ) (s ^ · • x)
  /-
    case intro.intro.intro
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite X
    x : X
    s : Subtype fun x => Membership.mem S x
    i j : Nat
    hne : Ne i j
    heq : Eq (HSMul.hSMul (HPow.hPow s i) x) (HSMul.hSMul (HPow.hPow s j) x)
    ⊢ Exists fun a => Eq ((fun x => OreLocalization.oreDiv x 1) a) (OreLocalizatio …
  -/
  wlog hlt : j < i generalizing i j
    /-
      case intro.intro.intro.inr
      R : Type u
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type v
      inst✝¹ : MulAction R X
      inst✝ : Finite X
      x : X
      s : Subtype fun x => Membership.mem S x
      i j : Nat
      hne : Ne i j
      heq : Eq (HSMul.hSMul (HPow.hPow s i) x) (HSMul.hSMul (HPow.hPow s j) x)
      this : ∀ (i j : Nat), Ne i j → Eq (HSMul.hSMul (HPow.hPow s i) x) (HSMul.hSMul …
      hlt : Not (LT.lt j i)
      ⊢ Exists fun a => Eq ((fun x => OreLocalization.oreDiv x 1) a) (OreLocalizatio …
    -/
  · exact this j i hne.symm heq.symm (hne.lt_of_le (not_lt.1 hlt))
    /-
      🎉 no goals
    -/
  /-
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite X
    x : X
    s : Subtype fun x => Membership.mem S x
    i j : Nat
    hne : Ne i j
    heq : Eq (HSMul.hSMul (HPow.hPow s i) x) (HSMul.hSMul (HPow.hPow s j) x)
    hlt : LT.lt j i
    ⊢ Exists fun a => Eq ((fun x => OreLocalization.oreDiv x 1) a) (OreLocalizatio …
  -/
  use s ^ (i - (j + 1)) • x
  /-
    case h
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite X
    x : X
    s : Subtype fun x => Membership.mem S x
    i j : Nat
    hne : Ne i j
    heq : Eq (HSMul.hSMul (HPow.hPow s i) x) (HSMul.hSMul (HPow.hPow s j) x)
    hlt : LT.lt j i
    ⊢ Eq ((fun x => OreLocalization.oreDiv x 1) (HSMul.hSMul (HPow.hPow s (HSub.hS …
  -/
  rw [oreDiv_eq_iff]
  /-
    case h
    R : Type u
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type v
    inst✝¹ : MulAction R X
    inst✝ : Finite X
    x : X
    s : Subtype fun x => Membership.mem S x
    i j : Nat
    hne : Ne i j
    heq : Eq (HSMul.hSMul (HPow.hPow s i) x) (HSMul.hSMul (HPow.hPow s j) x)
    hlt : LT.lt j i
    ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u x) (HSMul.hSMul v (HS …
  -/
  refine ⟨s ^ j, (s ^ (j + 1)).1, ?_, ?_⟩
    /-
      case h.refine_1
      R : Type u
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type v
      inst✝¹ : MulAction R X
      inst✝ : Finite X
      x : X
      s : Subtype fun x => Membership.mem S x
      i j : Nat
      hne : Ne i j
      heq : Eq (HSMul.hSMul (HPow.hPow s i) x) (HSMul.hSMul (HPow.hPow s j) x)
      hlt : LT.lt j i
      ⊢ Eq (HSMul.hSMul (HPow.hPow s j) x) (HSMul.hSMul (↑(HPow.hPow s (HAdd.hAdd j  …
    -/
  · change s ^ j • x = s ^ (j + 1) • s ^ (i - (j + 1)) • x
    /-
      case h.refine_1
      R : Type u
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type v
      inst✝¹ : MulAction R X
      inst✝ : Finite X
      x : X
      s : Subtype fun x => Membership.mem S x
      i j : Nat
      hne : Ne i j
      heq : Eq (HSMul.hSMul (HPow.hPow s i) x) (HSMul.hSMul (HPow.hPow s j) x)
      hlt : LT.lt j i
      ⊢ Eq (HSMul.hSMul (HPow.hPow s j) x) (HSMul.hSMul (HPow.hPow s (HAdd.hAdd j 1) …
    -/
    rw [← mul_smul, ← pow_add, Nat.add_sub_cancel' hlt, heq]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      R : Type u
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type v
      inst✝¹ : MulAction R X
      inst✝ : Finite X
      x : X
      s : Subtype fun x => Membership.mem S x
      i j : Nat
      hne : Ne i j
      heq : Eq (HSMul.hSMul (HPow.hPow s i) x) (HSMul.hSMul (HPow.hPow s j) x)
      hlt : LT.lt j i
      ⊢ Eq (HMul.hMul ↑(HPow.hPow s j) ↑s) (HMul.hMul ↑(HPow.hPow s (HAdd.hAdd j 1)) …
    -/
  · simp_rw [SubmonoidClass.coe_pow, OneMemClass.coe_one, mul_one, pow_succ]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem numeratorHom_surjective_of_finite [Finite S] : Surjective (numeratorHom (S := S)) :=
  oreDiv_one_surjective_of_finite_left S R


@[to_additive]
theorem cardinalMk_le_max : #(OreLocalization S X) ≤ max (lift.{v} #S) (lift.{u} #X) := by
  /-
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Max.max (Cardinal.lift.{v, u} (Ca …
  -/
  rcases finite_or_infinite X with _ | _
    /-
      case inl
      R : Type u
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type v
      inst✝ : MulAction R X
      h✝ : Finite X
      ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Max.max (Cardinal.lift.{v, u} (Ca …
    -/
  · have := lift_mk_le_lift_mk_of_surjective (oreDiv_one_surjective_of_finite_right S X)
    /-
      case inl
      R : Type u
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type v
      inst✝ : MulAction R X
      h✝ : Finite X
      this : LE.le (Cardinal.lift.{v, max u v} (Cardinal.mk (OreLocalization S X)))  …
      ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Max.max (Cardinal.lift.{v, u} (Ca …
    -/
    rw [lift_umax.{v, u}, lift_id'] at this
    /-
      case inl
      R : Type u
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type v
      inst✝ : MulAction R X
      h✝ : Finite X
      this : LE.le (Cardinal.mk (OreLocalization S X)) (Cardinal.lift.{u, v} (Cardin …
      ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Max.max (Cardinal.lift.{v, u} (Ca …
    -/
    exact le_max_of_le_right this
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    h✝ : Infinite X
    ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Max.max (Cardinal.lift.{v, u} (Ca …
  -/
  rcases finite_or_infinite S with _ | _
    /-
      case inr.inl
      R : Type u
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type v
      inst✝ : MulAction R X
      h✝¹ : Infinite X
      h✝ : Finite (Subtype fun x => Membership.mem S x)
      ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Max.max (Cardinal.lift.{v, u} (Ca …
    -/
  · have := lift_mk_le_lift_mk_of_surjective (oreDiv_one_surjective_of_finite_left S X)
    /-
      case inr.inl
      R : Type u
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type v
      inst✝ : MulAction R X
      h✝¹ : Infinite X
      h✝ : Finite (Subtype fun x => Membership.mem S x)
      this : LE.le (Cardinal.lift.{v, max u v} (Cardinal.mk (OreLocalization S X)))  …
      ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Max.max (Cardinal.lift.{v, u} (Ca …
    -/
    rw [lift_umax.{v, u}, lift_id'] at this
    /-
      case inr.inl
      R : Type u
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type v
      inst✝ : MulAction R X
      h✝¹ : Infinite X
      h✝ : Finite (Subtype fun x => Membership.mem S x)
      this : LE.le (Cardinal.mk (OreLocalization S X)) (Cardinal.lift.{u, v} (Cardin …
      ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Max.max (Cardinal.lift.{v, u} (Ca …
    -/
    exact le_max_of_le_right this
    /-
      🎉 no goals
    -/
  convert ← mk_le_of_surjective (show Surjective fun x : X × S ↦ x.1 /ₒ x.2 from
    Quotient.mk''_surjective)
  /-
    case h.e'_4
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    h✝¹ : Infinite X
    h✝ : Infinite (Subtype fun x => Membership.mem S x)
    ⊢ Eq (Cardinal.mk (Prod X (Subtype fun x => Membership.mem S x))) (Max.max (Ca …
  -/
  rw [mk_prod, mul_comm]
  /-
    case h.e'_4
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    h✝¹ : Infinite X
    h✝ : Infinite (Subtype fun x => Membership.mem S x)
    ⊢ Eq (HMul.hMul (Cardinal.lift.{v, u} (Cardinal.mk (Subtype fun x => Membershi …
  -/
                              /-
                                🎉 no goals
                              -/
  refine mul_eq_max ?_ ?_ <;> simp
                              /-
                                🎉 no goals
                              -/


@[to_additive]
theorem cardinalMk_le : #(OreLocalization S R) ≤ #R := by
  /-
    R : Type u
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    ⊢ LE.le (Cardinal.mk (OreLocalization S R)) (Cardinal.mk R)
  -/
  convert ← cardinalMk_le_max S R
  /-
    case h.e'_4
    R : Type u
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    ⊢ Eq (Max.max (Cardinal.lift.{u, u} (Cardinal.mk (Subtype fun x => Membership. …
  -/
  simp_rw [lift_id, max_eq_right_iff, mk_subtype_le]
  /-
    🎉 no goals
  -/

-- TODO: remove the `Commute` assumption

@[to_additive]
theorem cardinalMk_le_lift_cardinalMk_of_commute (hc : ∀ s s' : S, Commute s s') :
    #(OreLocalization S X) ≤ lift.{u} #X := by
  /-
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Cardinal.lift.{u, v} (Cardinal.mk …
  -/
  rcases finite_or_infinite X with _ | _
    /-
      case inl
      R : Type u
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type v
      inst✝ : MulAction R X
      hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
      h✝ : Finite X
      ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Cardinal.lift.{u, v} (Cardinal.mk …
    -/
  · have := lift_mk_le_lift_mk_of_surjective (oreDiv_one_surjective_of_finite_right S X)
    /-
      case inl
      R : Type u
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type v
      inst✝ : MulAction R X
      hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
      h✝ : Finite X
      this : LE.le (Cardinal.lift.{v, max u v} (Cardinal.mk (OreLocalization S X)))  …
      ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Cardinal.lift.{u, v} (Cardinal.mk …
    -/
    rwa [lift_umax.{v, u}, lift_id'] at this
    /-
      🎉 no goals
    -/
  have key (x : X) (s s' : S) (h : s • x = s' • x) (hc : Commute s s') : x /ₒ s = x /ₒ s' := by
    rw [oreDiv_eq_iff]
    refine ⟨s, s'.1, h, ?_⟩
    · exact_mod_cast hc
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Cardinal.lift.{u, v} (Cardinal.mk …
  -/
  let i (x : X × S) := x.1 /ₒ x.2
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Cardinal.lift.{u, v} (Cardinal.mk …
  -/
  have hsurj : Surjective i := Quotient.mk''_surjective
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    hsurj : Function.Surjective i
    ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Cardinal.lift.{u, v} (Cardinal.mk …
  -/
  have hi := rightInverse_surjInv hsurj
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    hsurj : Function.Surjective i
    hi : Function.RightInverse (Function.surjInv hsurj) i
    ⊢ LE.le (Cardinal.mk (OreLocalization S X)) (Cardinal.lift.{u, v} (Cardinal.mk …
  -/
  let j := (fun x : X × S ↦ (x.1, x.2 • x.1)) ∘ surjInv hsurj
  suffices Injective j by
    have := lift_mk_le_lift_mk_of_injective this
    rwa [lift_umax.{v, u}, lift_id', mk_prod, lift_id, lift_mul, mul_eq_self (by simp)] at this
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    hsurj : Function.Surjective i
    hi : Function.RightInverse (Function.surjInv hsurj) i
    j : OreLocalization S X → Prod X X := Function.comp (fun x => { fst := x.1, sn …
    ⊢ Function.Injective j
  -/
  intro y y' heq
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    hsurj : Function.Surjective i
    hi : Function.RightInverse (Function.surjInv hsurj) i
    j : OreLocalization S X → Prod X X := Function.comp (fun x => { fst := x.1, sn …
    y y' : OreLocalization S X
    heq : Eq (j y) (j y')
    ⊢ Eq y y'
  -/
  rw [← hi y, ← hi y']
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    hsurj : Function.Surjective i
    hi : Function.RightInverse (Function.surjInv hsurj) i
    j : OreLocalization S X → Prod X X := Function.comp (fun x => { fst := x.1, sn …
    y y' : OreLocalization S X
    heq : Eq (j y) (j y')
    ⊢ Eq (i (Function.surjInv hsurj y)) (i (Function.surjInv hsurj y'))
  -/
  simp_rw [j, comp_apply, Prod.ext_iff] at heq
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    hsurj : Function.Surjective i
    hi : Function.RightInverse (Function.surjInv hsurj) i
    j : OreLocalization S X → Prod X X := Function.comp (fun x => { fst := x.1, sn …
    y y' : OreLocalization S X
    heq : And (Eq (Function.surjInv hsurj y).1 (Function.surjInv hsurj y').1) (Eq  …
    ⊢ Eq (i (Function.surjInv hsurj y)) (i (Function.surjInv hsurj y'))
  -/
  simp_rw [i]
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    hsurj : Function.Surjective i
    hi : Function.RightInverse (Function.surjInv hsurj) i
    j : OreLocalization S X → Prod X X := Function.comp (fun x => { fst := x.1, sn …
    y y' : OreLocalization S X
    heq : And (Eq (Function.surjInv hsurj y).1 (Function.surjInv hsurj y').1) (Eq  …
    ⊢ Eq (OreLocalization.oreDiv (Function.surjInv hsurj y).1 (Function.surjInv hs …
  -/
  set x := surjInv hsurj y
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    hsurj : Function.Surjective i
    hi : Function.RightInverse (Function.surjInv hsurj) i
    j : OreLocalization S X → Prod X X := Function.comp (fun x => { fst := x.1, sn …
    y y' : OreLocalization S X
    x : Prod X (Subtype fun x => Membership.mem S x) := Function.surjInv hsurj y
    heq : And (Eq x.1 (Function.surjInv hsurj y').1) (Eq (HSMul.hSMul x.2 x.1) (HS …
    ⊢ Eq (OreLocalization.oreDiv x.1 x.2) (OreLocalization.oreDiv (Function.surjIn …
  -/
  set x' := surjInv hsurj y'
  /-
    case inr
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    hsurj : Function.Surjective i
    hi : Function.RightInverse (Function.surjInv hsurj) i
    j : OreLocalization S X → Prod X X := Function.comp (fun x => { fst := x.1, sn …
    y y' : OreLocalization S X
    x : Prod X (Subtype fun x => Membership.mem S x) := Function.surjInv hsurj y
    x' : Prod X (Subtype fun x => Membership.mem S x) := Function.surjInv hsurj y'
    heq : And (Eq x.1 x'.1) (Eq (HSMul.hSMul x.2 x.1) (HSMul.hSMul x'.2 x'.1))
    ⊢ Eq (OreLocalization.oreDiv x.1 x.2) (OreLocalization.oreDiv x'.1 x'.2)
  -/
  obtain ⟨h1, h2⟩ := heq
  /-
    case inr.intro
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    hsurj : Function.Surjective i
    hi : Function.RightInverse (Function.surjInv hsurj) i
    j : OreLocalization S X → Prod X X := Function.comp (fun x => { fst := x.1, sn …
    y y' : OreLocalization S X
    x : Prod X (Subtype fun x => Membership.mem S x) := Function.surjInv hsurj y
    x' : Prod X (Subtype fun x => Membership.mem S x) := Function.surjInv hsurj y'
    h1 : Eq x.1 x'.1
    h2 : Eq (HSMul.hSMul x.2 x.1) (HSMul.hSMul x'.2 x'.1)
    ⊢ Eq (OreLocalization.oreDiv x.1 x.2) (OreLocalization.oreDiv x'.1 x'.2)
  -/
  rw [← h1] at h2 ⊢
  /-
    case inr.intro
    R : Type u
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type v
    inst✝ : MulAction R X
    hc : ∀ (s s' : Subtype fun x => Membership.mem S x), Commute s s'
    h✝ : Infinite X
    key : ∀ (x : X) (s s' : Subtype fun x => Membership.mem S x), Eq (HSMul.hSMul  …
    i : Prod X (Subtype fun x => Membership.mem S x) → OreLocalization S X := fun  …
    hsurj : Function.Surjective i
    hi : Function.RightInverse (Function.surjInv hsurj) i
    j : OreLocalization S X → Prod X X := Function.comp (fun x => { fst := x.1, sn …
    y y' : OreLocalization S X
    x : Prod X (Subtype fun x => Membership.mem S x) := Function.surjInv hsurj y
    x' : Prod X (Subtype fun x => Membership.mem S x) := Function.surjInv hsurj y'
    h1 : Eq x.1 x'.1
    h2 : Eq (HSMul.hSMul x.2 x.1) (HSMul.hSMul x'.2 x.1)
    ⊢ Eq (OreLocalization.oreDiv x.1 x.2) (OreLocalization.oreDiv x.1 x'.2)
  -/
  exact key x.1 x.2 x'.2 h2 (hc _ _)
  /-
    🎉 no goals
  -/


