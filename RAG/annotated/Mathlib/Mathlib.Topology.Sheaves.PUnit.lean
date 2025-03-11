theorem isSheaf_of_isTerminal_of_indiscrete {X : TopCat.{w}} (hind : X.str = ⊤) (F : Presheaf C X)
    (it : IsTerminal <| F.obj <| op ⊥) : F.IsSheaf := fun c U s hs => by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    hind : Eq X.str Top.top
    F : TopCat.Presheaf C X
    it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
    c : C
    U : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Sieve U
    hs : Membership.mem ((Opens.grothendieckTopology ↑X) U) s
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.Functor.comp F (CategoryT …
  -/
  obtain rfl | hne := eq_or_ne U ⊥
    /-
      case inl
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      hind : Eq X.str Top.top
      F : TopCat.Presheaf C X
      it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
      c : C
      s : CategoryTheory.Sieve Bot.bot
      hs : Membership.mem ((Opens.grothendieckTopology ↑X) Bot.bot) s
      ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.Functor.comp F (CategoryT …
    -/
  · intro _ _
    /-
      case inl
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      hind : Eq X.str Top.top
      F : TopCat.Presheaf C X
      it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
      c : C
      s : CategoryTheory.Sieve Bot.bot
      hs : Membership.mem ((Opens.grothendieckTopology ↑X) Bot.bot) s
      x✝ : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.comp F ( …
      a✝ : x✝.Compatible
      ⊢ ExistsUnique fun t => x✝.IsAmalgamation t
    -/
    rw [@existsUnique_iff_exists _ ⟨fun _ _ => _⟩]
      /-
        case inl
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        hind : Eq X.str Top.top
        F : TopCat.Presheaf C X
        it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
        c : C
        s : CategoryTheory.Sieve Bot.bot
        hs : Membership.mem ((Opens.grothendieckTopology ↑X) Bot.bot) s
        x✝ : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.comp F ( …
        a✝ : x✝.Compatible
        ⊢ Exists fun x => x✝.IsAmalgamation x
      -/
    · refine ⟨it.from _, fun U hU hs => IsTerminal.hom_ext ?_ _ _⟩
      /-
        case inl
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        hind : Eq X.str Top.top
        F : TopCat.Presheaf C X
        it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
        c : C
        s : CategoryTheory.Sieve Bot.bot
        hs✝ : Membership.mem ((Opens.grothendieckTopology ↑X) Bot.bot) s
        x✝ : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.comp F ( …
        a✝ : x✝.Compatible
        U : TopologicalSpace.Opens ↑X
        hU : Quiver.Hom U Bot.bot
        hs : s.arrows hU
        ⊢ CategoryTheory.Limits.IsTerminal (F.obj { unop := U })
      -/
      rwa [le_bot_iff.1 hU.le]
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        hind : Eq X.str Top.top
        F : TopCat.Presheaf C X
        it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
        c : C
        s : CategoryTheory.Sieve Bot.bot
        hs : Membership.mem ((Opens.grothendieckTopology ↑X) Bot.bot) s
        x✝ : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.comp F ( …
        a✝ : x✝.Compatible
        ⊢ ∀ (x x_1 : (CategoryTheory.Functor.comp F (CategoryTheory.coyoneda.obj { uno …
      -/
    · apply it.hom_ext
      /-
        🎉 no goals
      -/
    /-
      case inr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      hind : Eq X.str Top.top
      F : TopCat.Presheaf C X
      it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
      c : C
      U : TopologicalSpace.Opens ↑X
      s : CategoryTheory.Sieve U
      hs : Membership.mem ((Opens.grothendieckTopology ↑X) U) s
      hne : Ne U Bot.bot
      ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.Functor.comp F (CategoryT …
    -/
  · convert Presieve.isSheafFor_top_sieve (F ⋙ coyoneda.obj (@op C c))
    /-
      case h.e'_5.h.h.e'_4
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      hind : Eq X.str Top.top
      F : TopCat.Presheaf C X
      it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
      c : C
      U : TopologicalSpace.Opens ↑X
      s : CategoryTheory.Sieve U
      hs : Membership.mem ((Opens.grothendieckTopology ↑X) U) s
      hne : Ne U Bot.bot
      ⊢ Eq s Top.top
    -/
    rw [← Sieve.id_mem_iff_eq_top]
    /-
      case h.e'_5.h.h.e'_4
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      hind : Eq X.str Top.top
      F : TopCat.Presheaf C X
      it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
      c : C
      U : TopologicalSpace.Opens ↑X
      s : CategoryTheory.Sieve U
      hs : Membership.mem ((Opens.grothendieckTopology ↑X) U) s
      hne : Ne U Bot.bot
      ⊢ s.arrows (CategoryTheory.CategoryStruct.id U)
    -/
    have := (U.eq_bot_or_top hind).resolve_left hne
    /-
      case h.e'_5.h.h.e'_4
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      hind : Eq X.str Top.top
      F : TopCat.Presheaf C X
      it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
      c : C
      U : TopologicalSpace.Opens ↑X
      s : CategoryTheory.Sieve U
      hs : Membership.mem ((Opens.grothendieckTopology ↑X) U) s
      hne : Ne U Bot.bot
      this : Eq U Top.top
      ⊢ s.arrows (CategoryTheory.CategoryStruct.id U)
    -/
    subst this
    /-
      case h.e'_5.h.h.e'_4
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      hind : Eq X.str Top.top
      F : TopCat.Presheaf C X
      it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
      c : C
      s : CategoryTheory.Sieve Top.top
      hs : Membership.mem ((Opens.grothendieckTopology ↑X) Top.top) s
      hne : Ne Top.top Bot.bot
      ⊢ s.arrows (CategoryTheory.CategoryStruct.id Top.top)
    -/
    obtain he | ⟨⟨x⟩⟩ := isEmpty_or_nonempty X
      /-
        case h.e'_5.h.h.e'_4.inl
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        hind : Eq X.str Top.top
        F : TopCat.Presheaf C X
        it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
        c : C
        s : CategoryTheory.Sieve Top.top
        hs : Membership.mem ((Opens.grothendieckTopology ↑X) Top.top) s
        hne : Ne Top.top Bot.bot
        he : IsEmpty ↑X
        ⊢ s.arrows (CategoryTheory.CategoryStruct.id Top.top)
      -/
    · exact (hne <| SetLike.ext'_iff.2 <| Set.univ_eq_empty_iff.2 he).elim
      /-
        🎉 no goals
      -/
    /-
      case h.e'_5.h.h.e'_4.inr.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      hind : Eq X.str Top.top
      F : TopCat.Presheaf C X
      it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
      c : C
      s : CategoryTheory.Sieve Top.top
      hs : Membership.mem ((Opens.grothendieckTopology ↑X) Top.top) s
      hne : Ne Top.top Bot.bot
      x : ↑X
      ⊢ s.arrows (CategoryTheory.CategoryStruct.id Top.top)
    -/
    obtain ⟨U, f, hf, hm⟩ := hs x _root_.trivial
    /-
      case h.e'_5.h.h.e'_4.inr.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      hind : Eq X.str Top.top
      F : TopCat.Presheaf C X
      it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
      c : C
      s : CategoryTheory.Sieve Top.top
      hs : Membership.mem ((Opens.grothendieckTopology ↑X) Top.top) s
      hne : Ne Top.top Bot.bot
      x : ↑X
      U : TopologicalSpace.Opens ↑X
      f : Quiver.Hom U Top.top
      hf : s.arrows f
      hm : Membership.mem U x
      ⊢ s.arrows (CategoryTheory.CategoryStruct.id Top.top)
    -/
    obtain rfl | rfl := U.eq_bot_or_top hind
      /-
        case h.e'_5.h.h.e'_4.inr.intro.intro.intro.intro.inl
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        hind : Eq X.str Top.top
        F : TopCat.Presheaf C X
        it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
        c : C
        s : CategoryTheory.Sieve Top.top
        hs : Membership.mem ((Opens.grothendieckTopology ↑X) Top.top) s
        hne : Ne Top.top Bot.bot
        x : ↑X
        f : Quiver.Hom Bot.bot Top.top
        hf : s.arrows f
        hm : Membership.mem Bot.bot x
        ⊢ s.arrows (CategoryTheory.CategoryStruct.id Top.top)
      -/
    · cases hm
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.h.h.e'_4.inr.intro.intro.intro.intro.inr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        hind : Eq X.str Top.top
        F : TopCat.Presheaf C X
        it : CategoryTheory.Limits.IsTerminal (F.obj { unop := Bot.bot })
        c : C
        s : CategoryTheory.Sieve Top.top
        hs : Membership.mem ((Opens.grothendieckTopology ↑X) Top.top) s
        hne : Ne Top.top Bot.bot
        x : ↑X
        f : Quiver.Hom Top.top Top.top
        hf : s.arrows f
        hm : Membership.mem Top.top x
        ⊢ s.arrows (CategoryTheory.CategoryStruct.id Top.top)
      -/
    · convert hf
      /-
        🎉 no goals
      -/


theorem isSheaf_iff_isTerminal_of_indiscrete {X : TopCat.{w}} (hind : X.str = ⊤)
    (F : Presheaf C X) : F.IsSheaf ↔ Nonempty (IsTerminal <| F.obj <| op ⊥) :=
  ⟨fun h => ⟨Sheaf.isTerminalOfEmpty ⟨F, h⟩⟩, fun ⟨it⟩ =>
    isSheaf_of_isTerminal_of_indiscrete hind F it⟩


theorem isSheaf_on_punit_of_isTerminal (F : Presheaf C (TopCat.of PUnit))
    (it : IsTerminal <| F.obj <| op ⊥) : F.IsSheaf :=
  isSheaf_of_isTerminal_of_indiscrete (@Subsingleton.elim (TopologicalSpace PUnit) _ _ _) F it


theorem isSheaf_on_punit_iff_isTerminal (F : Presheaf C (TopCat.of PUnit)) :
    F.IsSheaf ↔ Nonempty (IsTerminal <| F.obj <| op ⊥) :=
  ⟨fun h => ⟨Sheaf.isTerminalOfEmpty ⟨F, h⟩⟩, fun ⟨it⟩ => isSheaf_on_punit_of_isTerminal F it⟩


