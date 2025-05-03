/--
The kernel of a ring homomorphism, as a two-sided ideal.
-/
def ker : TwoSidedIdeal R :=
  .mk
  { r := fun x y ↦ f x = f y
                /-
                  R : Type u_1
                  S : Type u_2
                  inst✝³ : NonUnitalNonAssocRing R
                  inst✝² : NonUnitalNonAssocSemiring S
                  F : Type u_3
                  inst✝¹ : FunLike F R S
                  inst✝ : NonUnitalRingHomClass F R S
                  f : F
                  ⊢ Equivalence fun x y => Eq (f x) (f y)
                -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
    iseqv := by constructor <;> aesop
                                /-
                                  🎉 no goals
                                -/
               /-
                 R : Type u_1
                 S : Type u_2
                 inst✝³ : NonUnitalNonAssocRing R
                 inst✝² : NonUnitalNonAssocSemiring S
                 F : Type u_3
                 inst✝¹ : FunLike F R S
                 inst✝ : NonUnitalRingHomClass F R S
                 f : F
                 ⊢ ∀ {w x y z : R}, { r := fun x y => Eq (f x) (f y), iseqv := ⋯ } w x → { r := …
               -/
    mul' := by intro; simp_all [map_add]
                      /-
                        🎉 no goals
                      -/
               /-
                 R : Type u_1
                 S : Type u_2
                 inst✝³ : NonUnitalNonAssocRing R
                 inst✝² : NonUnitalNonAssocSemiring S
                 F : Type u_3
                 inst✝¹ : FunLike F R S
                 inst✝ : NonUnitalRingHomClass F R S
                 f : F
                 ⊢ ∀ {w x y z : R}, { r := fun x y => Eq (f x) (f y), iseqv := ⋯, mul' := ⋯ }.t …
               -/
    add' := by intro; simp_all [map_mul] }
                      /-
                        🎉 no goals
                      -/


@[simp]
lemma ker_ringCon {x y : R} : (ker f).ringCon x y ↔ f x = f y := Iff.rfl


lemma mem_ker {x : R} : x ∈ ker f ↔ f x = 0 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : NonUnitalNonAssocRing R
    inst✝² : NonUnitalNonAssocSemiring S
    F : Type u_3
    inst✝¹ : FunLike F R S
    inst✝ : NonUnitalRingHomClass F R S
    f : F
    x : R
    ⊢ Iff (Membership.mem (TwoSidedIdeal.ker f) x) (Eq (f x) 0)
  -/
  rw [mem_iff, ker_ringCon, map_zero]
  /-
    🎉 no goals
  -/


lemma ker_eq_bot : ker f = ⊥ ↔ Function.Injective f := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : NonUnitalNonAssocRing R
    inst✝² : NonUnitalNonAssocSemiring S
    F : Type u_3
    inst✝¹ : FunLike F R S
    inst✝ : NonUnitalRingHomClass F R S
    f : F
    ⊢ Iff (Eq (TwoSidedIdeal.ker f) Bot.bot) (Function.Injective ⇑f)
  -/
  fconstructor
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝³ : NonUnitalNonAssocRing R
      inst✝² : NonUnitalNonAssocSemiring S
      F : Type u_3
      inst✝¹ : FunLike F R S
      inst✝ : NonUnitalRingHomClass F R S
      f : F
      ⊢ Eq (TwoSidedIdeal.ker f) Bot.bot → Function.Injective ⇑f
    -/
  · intro h x y hxy
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝³ : NonUnitalNonAssocRing R
      inst✝² : NonUnitalNonAssocSemiring S
      F : Type u_3
      inst✝¹ : FunLike F R S
      inst✝ : NonUnitalRingHomClass F R S
      f : F
      h : Eq (TwoSidedIdeal.ker f) Bot.bot
      x y : R
      hxy : Eq (f x) (f y)
      ⊢ Eq x y
    -/
    simpa [h, rel_iff, mem_bot, sub_eq_zero] using show (ker f).ringCon x y from hxy
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      S : Type u_2
      inst✝³ : NonUnitalNonAssocRing R
      inst✝² : NonUnitalNonAssocSemiring S
      F : Type u_3
      inst✝¹ : FunLike F R S
      inst✝ : NonUnitalRingHomClass F R S
      f : F
      ⊢ Function.Injective ⇑f → Eq (TwoSidedIdeal.ker f) Bot.bot
    -/
  · exact fun h ↦ eq_bot_iff.2 fun x hx => h hx
    /-
      🎉 no goals
    -/


/--
The kernel of the ring homomorphism `R → R⧸I` is `I`.
-/
@[simp]
lemma ker_ringCon_mk' (I : TwoSidedIdeal R) : ker I.ringCon.mk' = I :=
  le_antisymm
                   /-
                     R : Type u_4
                     inst✝ : NonAssocRing R
                     I : TwoSidedIdeal R
                     x✝ : R
                     h : Membership.mem (TwoSidedIdeal.ker I.ringCon.mk') x✝
                     ⊢ Membership.mem I x✝
                   -/
    (fun _ h => by simpa using I.rel_iff _ _ |>.1 (Quotient.eq'.1 h))
                   /-
                     🎉 no goals
                   -/
                                                          /-
                                                            R : Type u_4
                                                            inst✝ : NonAssocRing R
                                                            I : TwoSidedIdeal R
                                                            x✝ : R
                                                            h : Membership.mem I x✝
                                                            ⊢ Membership.mem I (HSub.hSub x✝ 0)
                                                          -/
    (fun _ h => Quotient.sound' <| I.rel_iff _ _ |>.2 (by simpa using h))
                                                          /-
                                                            🎉 no goals
                                                          -/


