theorem of_isUnit_or_isUnit_of_isUnit_add [Nontrivial R]
    (h : ∀ a b : R, IsUnit (a + b) → IsUnit a ∨ IsUnit b) : IsLocalRing R :=
  ⟨fun {a b} hab => h a b <| hab.symm ▸ isUnit_one⟩


/-- A semiring is local if it is nontrivial and the set of nonunits is closed under the addition. -/
theorem of_nonunits_add [Nontrivial R]
    (h : ∀ a b : R, a ∈ nonunits R → b ∈ nonunits R → a + b ∈ nonunits R) : IsLocalRing R where
  isUnit_or_isUnit_of_add_one {a b} hab :=
    or_iff_not_and_not.2 fun H => h a b H.1 H.2 <| hab.symm ▸ isUnit_one


/-- A semiring is local if it has a unique maximal ideal. -/
theorem of_unique_max_ideal (h : ∃! I : Ideal R, I.IsMaximal) : IsLocalRing R :=
  @of_nonunits_add _ _
    (nontrivial_of_ne (0 : R) 1 <|
      let ⟨I, Imax, _⟩ := h
      fun H : 0 = 1 => Imax.1.1 <| I.eq_top_iff_one.2 <| H ▸ I.zero_mem)
    fun x y hx hy H =>
    let ⟨I, Imax, Iuniq⟩ := h
    let ⟨Ix, Ixmax, Hx⟩ := exists_max_ideal_of_mem_nonunits hx
    let ⟨Iy, Iymax, Hy⟩ := exists_max_ideal_of_mem_nonunits hy
    have xmemI : x ∈ I := Iuniq Ix Ixmax ▸ Hx
    have ymemI : y ∈ I := Iuniq Iy Iymax ▸ Hy
    Imax.1.1 <| I.eq_top_of_isUnit_mem (I.add_mem xmemI ymemI) H


theorem of_unique_nonzero_prime (h : ∃! P : Ideal R, P ≠ ⊥ ∧ Ideal.IsPrime P) : IsLocalRing R :=
  of_unique_max_ideal
    (by
      /-
        R : Type u_1
        inst✝ : CommSemiring R
        h : ExistsUnique fun P => And (Ne P Bot.bot) P.IsPrime
        ⊢ ExistsUnique fun I => I.IsMaximal
      -/
      rcases h with ⟨P, ⟨hPnonzero, hPnot_top, _⟩, hPunique⟩
      /-
        case intro.intro.intro.mk
        R : Type u_1
        inst✝ : CommSemiring R
        P : Ideal R
        hPunique : ∀ (y : Ideal R), (fun P => And (Ne P Bot.bot) P.IsPrime) y → Eq y P
        hPnonzero : Ne P Bot.bot
        hPnot_top : Ne P Top.top
        mem_or_mem'✝ : ∀ {x y : R}, Membership.mem P (HMul.hMul x y) → Or (Membership. …
        ⊢ ExistsUnique fun I => I.IsMaximal
      -/
      refine ⟨P, ⟨⟨hPnot_top, ?_⟩⟩, fun M hM => hPunique _ ⟨?_, Ideal.IsMaximal.isPrime hM⟩⟩
        /-
          case intro.intro.intro.mk.refine_1
          R : Type u_1
          inst✝ : CommSemiring R
          P : Ideal R
          hPunique : ∀ (y : Ideal R), (fun P => And (Ne P Bot.bot) P.IsPrime) y → Eq y P
          hPnonzero : Ne P Bot.bot
          hPnot_top : Ne P Top.top
          mem_or_mem'✝ : ∀ {x y : R}, Membership.mem P (HMul.hMul x y) → Or (Membership. …
          ⊢ ∀ (b : Ideal R), LT.lt P b → Eq b Top.top
        -/
      · refine Ideal.maximal_of_no_maximal fun M hPM hM => ne_of_lt hPM ?_
        /-
          case intro.intro.intro.mk.refine_1
          R : Type u_1
          inst✝ : CommSemiring R
          P : Ideal R
          hPunique : ∀ (y : Ideal R), (fun P => And (Ne P Bot.bot) P.IsPrime) y → Eq y P
          hPnonzero : Ne P Bot.bot
          hPnot_top : Ne P Top.top
          mem_or_mem'✝ : ∀ {x y : R}, Membership.mem P (HMul.hMul x y) → Or (Membership. …
          M : Ideal R
          hPM : LT.lt P M
          hM : M.IsMaximal
          ⊢ Eq P M
        -/
        exact (hPunique _ ⟨ne_bot_of_gt hPM, Ideal.IsMaximal.isPrime hM⟩).symm
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.mk.refine_2
          R : Type u_1
          inst✝ : CommSemiring R
          P : Ideal R
          hPunique : ∀ (y : Ideal R), (fun P => And (Ne P Bot.bot) P.IsPrime) y → Eq y P
          hPnonzero : Ne P Bot.bot
          hPnot_top : Ne P Top.top
          mem_or_mem'✝ : ∀ {x y : R}, Membership.mem P (HMul.hMul x y) → Or (Membership. …
          M : Ideal R
          hM : (fun I => I.IsMaximal) M
          ⊢ Ne M Bot.bot
        -/
      · rintro rfl
        /-
          case intro.intro.intro.mk.refine_2
          R : Type u_1
          inst✝ : CommSemiring R
          P : Ideal R
          hPunique : ∀ (y : Ideal R), (fun P => And (Ne P Bot.bot) P.IsPrime) y → Eq y P
          hPnonzero : Ne P Bot.bot
          hPnot_top : Ne P Top.top
          mem_or_mem'✝ : ∀ {x y : R}, Membership.mem P (HMul.hMul x y) → Or (Membership. …
          hM : Bot.bot.IsMaximal
          ⊢ False
        -/
        exact hPnot_top (hM.1.2 P (bot_lt_iff_ne_bot.2 hPnonzero)))
        /-
          🎉 no goals
        -/


theorem isUnit_or_isUnit_of_isUnit_add {a b : R} (h : IsUnit (a + b)) : IsUnit a ∨ IsUnit b := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : IsLocalRing R
    a b : R
    h : IsUnit (HAdd.hAdd a b)
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  rcases h with ⟨u, hu⟩
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : IsLocalRing R
    a b : R
    u : Units R
    hu : Eq (↑u) (HAdd.hAdd a b)
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  rw [← Units.inv_mul_eq_one, mul_add] at hu
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : IsLocalRing R
    a b : R
    u : Units R
    hu : Eq (HAdd.hAdd (HMul.hMul (↑(Inv.inv u)) a) (HMul.hMul (↑(Inv.inv u)) b)) 1
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  apply Or.imp _ _ (isUnit_or_isUnit_of_add_one hu) <;> exact isUnit_of_mul_isUnit_right
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem nonunits_add {a b : R} (ha : a ∈ nonunits R) (hb : b ∈ nonunits R) : a + b ∈ nonunits R :=
  fun H => not_or_intro ha hb (isUnit_or_isUnit_of_isUnit_add H)


@[deprecated (since := "2024-11-11")]
alias LocalRing.of_isUnit_or_isUnit_of_isUnit_add := IsLocalRing.of_isUnit_or_isUnit_of_isUnit_add


@[deprecated (since := "2024-11-11")]
alias LocalRing.of_nonunits_add := IsLocalRing.of_nonunits_add


@[deprecated (since := "2024-11-11")]
alias LocalRing.of_unique_max_ideal := IsLocalRing.of_unique_max_ideal


@[deprecated (since := "2024-11-11")]
alias LocalRing.of_unique_nonzero_prime := IsLocalRing.of_unique_nonzero_prime


@[deprecated (since := "2024-11-11")]
alias LocalRing.isUnit_or_isUnit_of_isUnit_add := IsLocalRing.isUnit_or_isUnit_of_isUnit_add


@[deprecated (since := "2024-11-11")]
alias LocalRing.nonunits_add := IsLocalRing.nonunits_add


theorem of_isUnit_or_isUnit_one_sub_self [Nontrivial R] (h : ∀ a : R, IsUnit a ∨ IsUnit (1 - a)) :
    IsLocalRing R :=
  ⟨fun {a b} hab => add_sub_cancel_left a b ▸ hab.symm ▸ h a⟩


theorem isUnit_or_isUnit_one_sub_self (a : R) : IsUnit a ∨ IsUnit (1 - a) :=
  isUnit_or_isUnit_of_isUnit_add <| (add_sub_cancel a 1).symm ▸ isUnit_one


theorem isUnit_of_mem_nonunits_one_sub_self (a : R) (h : 1 - a ∈ nonunits R) : IsUnit a :=
  or_iff_not_imp_right.1 (isUnit_or_isUnit_one_sub_self a) h


theorem isUnit_one_sub_self_of_mem_nonunits (a : R) (h : a ∈ nonunits R) : IsUnit (1 - a) :=
  or_iff_not_imp_left.1 (isUnit_or_isUnit_one_sub_self a) h


theorem of_surjective' [Ring S] [Nontrivial S] (f : R →+* S) (hf : Function.Surjective f) :
    IsLocalRing S :=
  of_isUnit_or_isUnit_one_sub_self (by
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsLocalRing R
      inst✝¹ : Ring S
      inst✝ : Nontrivial S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      ⊢ ∀ (a : S), Or (IsUnit a) (IsUnit (HSub.hSub 1 a))
    -/
    intro b
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsLocalRing R
      inst✝¹ : Ring S
      inst✝ : Nontrivial S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      b : S
      ⊢ Or (IsUnit b) (IsUnit (HSub.hSub 1 b))
    -/
    obtain ⟨a, rfl⟩ := hf b
    /-
      case intro
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsLocalRing R
      inst✝¹ : Ring S
      inst✝ : Nontrivial S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      a : R
      ⊢ Or (IsUnit (f a)) (IsUnit (HSub.hSub 1 (f a)))
    -/
    apply (isUnit_or_isUnit_one_sub_self a).imp <| RingHom.isUnit_map _
    /-
      case intro
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsLocalRing R
      inst✝¹ : Ring S
      inst✝ : Nontrivial S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      a : R
      ⊢ IsUnit (HSub.hSub 1 a) → IsUnit (HSub.hSub 1 (f a))
    -/
    rw [← f.map_one, ← f.map_sub]
    /-
      case intro
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsLocalRing R
      inst✝¹ : Ring S
      inst✝ : Nontrivial S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      a : R
      ⊢ IsUnit (HSub.hSub 1 a) → IsUnit (f (HSub.hSub 1 a))
    -/
    apply f.isUnit_map)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-11")]
alias LocalRing.of_isUnit_or_isUnit_one_sub_self := IsLocalRing.of_isUnit_or_isUnit_one_sub_self


@[deprecated (since := "2024-11-11")]
alias LocalRing.isUnit_or_isUnit_one_sub_self := IsLocalRing.isUnit_or_isUnit_one_sub_self


@[deprecated (since := "2024-11-11")]
alias LocalRing.isUnit_of_mem_nonunits_one_sub_self :=
  IsLocalRing.isUnit_of_mem_nonunits_one_sub_self


@[deprecated (since := "2024-11-11")]
alias LocalRing.isUnit_one_sub_self_of_mem_nonunits :=
  IsLocalRing.isUnit_one_sub_self_of_mem_nonunits


@[deprecated (since := "2024-11-11")]
alias LocalRing.of_surjective' := IsLocalRing.of_surjective'


instance (priority := 100) : IsLocalRing K := by
  classical exact IsLocalRing.of_isUnit_or_isUnit_one_sub_self fun a =>
    if h : a = 0 then Or.inr (by rw [h, sub_zero]; exact isUnit_one)
    else Or.inl <| IsUnit.mk0 a h


