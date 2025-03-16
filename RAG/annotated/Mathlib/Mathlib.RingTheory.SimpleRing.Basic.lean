instance [IsSimpleRing R] : IsSimpleOrder (TwoSidedIdeal R) := IsSimpleRing.simple


instance [simple : IsSimpleRing R] : Nontrivial R := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    simple : IsSimpleRing R
    ⊢ Nontrivial R
  -/
  obtain ⟨x, hx⟩ := SetLike.exists_of_lt (bot_lt_top : (⊥ : TwoSidedIdeal R) < ⊤)
  /-
    case intro
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    simple : IsSimpleRing R
    x : R
    hx : And (Membership.mem Top.top x) (Not (Membership.mem Bot.bot x))
    ⊢ Nontrivial R
  -/
  have h (hx : x = 0) : False := by simp_all [TwoSidedIdeal.zero_mem]
  /-
    case intro
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    simple : IsSimpleRing R
    x : R
    hx : And (Membership.mem Top.top x) (Not (Membership.mem Bot.bot x))
    h : Eq x 0 → False
    ⊢ Nontrivial R
  -/
  use x, 0, h
  /-
    🎉 no goals
  -/


lemma one_mem_of_ne_bot {A : Type*} [NonAssocRing A] [IsSimpleRing A] (I : TwoSidedIdeal A)
    (hI : I ≠ ⊥) : (1 : A) ∈ I :=
  (eq_bot_or_eq_top I).resolve_left hI ▸ ⟨⟩


lemma one_mem_of_ne_zero_mem {A : Type*} [NonAssocRing A] [IsSimpleRing A] (I : TwoSidedIdeal A)
    {x : A} (hx : x ≠ 0) (hxI : x ∈ I) : (1 : A) ∈ I :=
                          /-
                            A : Type u_2
                            inst✝¹ : NonAssocRing A
                            inst✝ : IsSimpleRing A
                            I : TwoSidedIdeal A
                            x : A
                            hx : Ne x 0
                            hxI : Membership.mem I x
                            ⊢ Ne I Bot.bot
                          -/
  one_mem_of_ne_bot I (by rintro rfl; exact hx hxI)
                                      /-
                                        🎉 no goals
                                      -/


lemma of_eq_bot_or_eq_top [Nontrivial R] (h : ∀ I : TwoSidedIdeal R, I = ⊥ ∨ I = ⊤) :
    IsSimpleRing R where
  simple := { eq_bot_or_eq_top := h }


instance _root_.DivisionRing.isSimpleRing (A : Type*) [DivisionRing A] : IsSimpleRing A :=
  .of_eq_bot_or_eq_top <| fun I ↦ by
    /-
      R : Type u_1
      inst✝¹ : NonUnitalNonAssocRing R
      A : Type u_2
      inst✝ : DivisionRing A
      I : TwoSidedIdeal A
      ⊢ Or (Eq I Bot.bot) (Eq I Top.top)
    -/
    rw [or_iff_not_imp_left, ← I.one_mem_iff]
    /-
      R : Type u_1
      inst✝¹ : NonUnitalNonAssocRing R
      A : Type u_2
      inst✝ : DivisionRing A
      I : TwoSidedIdeal A
      ⊢ Not (Eq I Bot.bot) → Membership.mem I 1
    -/
    intro H
    /-
      R : Type u_1
      inst✝¹ : NonUnitalNonAssocRing R
      A : Type u_2
      inst✝ : DivisionRing A
      I : TwoSidedIdeal A
      H : Not (Eq I Bot.bot)
      ⊢ Membership.mem I 1
    -/
    obtain ⟨x, hx1, hx2 : x ≠ 0⟩ := SetLike.exists_of_lt (bot_lt_iff_ne_bot.mpr H : ⊥ < I)
    /-
      case intro.intro
      R : Type u_1
      inst✝¹ : NonUnitalNonAssocRing R
      A : Type u_2
      inst✝ : DivisionRing A
      I : TwoSidedIdeal A
      H : Not (Eq I Bot.bot)
      x : A
      hx1 : Membership.mem I x
      hx2 : Ne x 0
      ⊢ Membership.mem I 1
    -/
    simpa [inv_mul_cancel₀ hx2] using I.mul_mem_left x⁻¹ _ hx1
    /-
      🎉 no goals
    -/


lemma injective_ringHom_or_subsingleton_codomain
    {R S : Type*} [NonAssocRing R] [IsSimpleRing R] [NonAssocSemiring S]
    (f : R →+* S) : Function.Injective f ∨ Subsingleton S :=
  simple.eq_bot_or_eq_top (TwoSidedIdeal.ker f) |>.imp (TwoSidedIdeal.ker_eq_bot _ |>.1)
    (fun h => subsingleton_iff_zero_eq_one.1 <| by
      /-
        R : Type u_2
        S : Type u_3
        inst✝² : NonAssocRing R
        inst✝¹ : IsSimpleRing R
        inst✝ : NonAssocSemiring S
        f : RingHom R S
        h : Eq (TwoSidedIdeal.ker f) Top.top
        ⊢ Eq 0 1
      -/
      have mem : 1 ∈ TwoSidedIdeal.ker f := h.symm ▸ TwoSidedIdeal.mem_top _
      /-
        R : Type u_2
        S : Type u_3
        inst✝² : NonAssocRing R
        inst✝¹ : IsSimpleRing R
        inst✝ : NonAssocSemiring S
        f : RingHom R S
        h : Eq (TwoSidedIdeal.ker f) Top.top
        mem : Membership.mem (TwoSidedIdeal.ker f) 1
        ⊢ Eq 0 1
      -/
      rwa [TwoSidedIdeal.mem_ker, map_one, eq_comm] at mem)
      /-
        🎉 no goals
      -/


protected theorem _root_.RingHom.injective
    {R S : Type*} [NonAssocRing R] [IsSimpleRing R] [NonAssocSemiring S] [Nontrivial S]
    (f : R →+* S) : Function.Injective f :=
  injective_ringHom_or_subsingleton_codomain f |>.resolve_right fun r => not_subsingleton _ r


universe u in
lemma iff_injective_ringHom_or_subsingleton_codomain (R : Type u) [NonAssocRing R] [Nontrivial R] :
    IsSimpleRing R ↔
    ∀ {S : Type u} [NonAssocSemiring S] (f : R →+* S), Function.Injective f ∨ Subsingleton S where
  mp _ _ _ := injective_ringHom_or_subsingleton_codomain
  mpr H := of_eq_bot_or_eq_top fun I => H I.ringCon.mk' |>.imp
    (fun h => le_antisymm
      (fun _ hx => TwoSidedIdeal.ker_eq_bot _ |>.2 h ▸ I.ker_ringCon_mk'.symm ▸ hx) bot_le)
    (fun h => le_antisymm le_top fun x _ => I.mem_iff _ |>.2 (Quotient.eq'.1 (h.elim x 0)))


universe u in
lemma iff_injective_ringHom (R : Type u) [NonAssocRing R] [Nontrivial R] :
    IsSimpleRing R ↔
    ∀ {S : Type u} [NonAssocSemiring S] [Nontrivial S] (f : R →+* S), Function.Injective f :=
  iff_injective_ringHom_or_subsingleton_codomain R |>.trans <|
                                               /-
                                                 R : Type u
                                                 inst✝¹ : NonAssocRing R
                                                 inst✝ : Nontrivial R
                                                 H : ∀ {S : Type u} [inst : NonAssocSemiring S] (f : RingHom R S), Or (Function …
                                                 x✝² : Type u
                                                 x✝¹ : NonAssocSemiring x✝²
                                                 x✝ : Nontrivial x✝²
                                                 f : RingHom R x✝²
                                                 ⊢ Not (Subsingleton x✝²)
                                               -/
    ⟨fun H _ _ _ f => H f |>.resolve_right (by simpa [not_subsingleton_iff_nontrivial]),
                                               /-
                                                 🎉 no goals
                                               -/
      fun H S _ f => subsingleton_or_nontrivial S |>.recOn Or.inr fun _ => Or.inl <| H f⟩


