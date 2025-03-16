/-- A convenience formulation for a pushout being a van Kampen colimit.
See `IsPushout.isVanKampen_iff` below. -/
@[nolint unusedArguments]
def IsPushout.IsVanKampen (_ : IsPushout f g h i) : Prop :=
  ∀ ⦃W' X' Y' Z' : C⦄ (f' : W' ⟶ X') (g' : W' ⟶ Y') (h' : X' ⟶ Z') (i' : Y' ⟶ Z') (αW : W' ⟶ W)
    (αX : X' ⟶ X) (αY : Y' ⟶ Y) (αZ : Z' ⟶ Z) (_ : IsPullback f' αW αX f)
    (_ : IsPullback g' αW αY g) (_ : CommSq h' αX αZ h) (_ : CommSq i' αY αZ i)
    (_ : CommSq f' g' h' i'), IsPushout f' g' h' i' ↔ IsPullback h' αX αZ h ∧ IsPullback i' αY αZ i


theorem IsPushout.IsVanKampen.flip {H : IsPushout f g h i} (H' : H.IsVanKampen) :
    H.flip.IsVanKampen := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    H : CategoryTheory.IsPushout f g h i
    H' : H.IsVanKampen
    ⊢ ⋯.IsVanKampen
  -/
  introv W' hf hg hh hi w
  simpa only [IsPushout.flip_iff, IsPullback.flip_iff, and_comm] using
    H' g' f' i' h' αW αY αX αZ hg hf hi hh w.flip


theorem IsPushout.isVanKampen_iff (H : IsPushout f g h i) :
    H.IsVanKampen ↔ IsVanKampenColimit (PushoutCocone.mk h i H.w) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    H : CategoryTheory.IsPushout f g h i
    ⊢ Iff H.IsVanKampen (CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits. …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      H : CategoryTheory.IsPushout f g h i
      ⊢ H.IsVanKampen → CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.Pus …
    -/
  · intro H F' c' α fα eα hα
    refine Iff.trans ?_
        ((H (F'.map WalkingSpan.Hom.fst) (F'.map WalkingSpan.Hom.snd) (c'.ι.app _) (c'.ι.app _)
          (α.app _) (α.app _) (α.app _) fα (by convert hα WalkingSpan.Hom.fst)
          (by convert hα WalkingSpan.Hom.snd) ?_ ?_ ?_).trans ?_)
    · have : F'.map WalkingSpan.Hom.fst ≫ c'.ι.app WalkingSpan.left =
          F'.map WalkingSpan.Hom.snd ≫ c'.ι.app WalkingSpan.right := by
        simp only [Cocone.w]
      rw [(IsColimit.equivOfNatIsoOfIso (diagramIsoSpan F') c' (PushoutCocone.mk _ _ this)
            _).nonempty_congr]
        /-
          case mp.refine_1
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          h : Quiver.Hom X Z
          i : Quiver.Hom Y Z
          H✝ : CategoryTheory.IsPushout f g h i
          H : H✝.IsVanKampen
          F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
          c' : CategoryTheory.Limits.Cocone F'
          α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
          fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
          eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
          hα : CategoryTheory.NatTrans.Equifibered α
          this : Eq (CategoryTheory.CategoryStruct.comp (F'.map CategoryTheory.Limits.Wa …
          ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Pushou …
        -/
      · exact ⟨fun h => ⟨⟨this⟩, h⟩, fun h => h.2⟩
        /-
          🎉 no goals
        -/
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          h : Quiver.Hom X Z
          i : Quiver.Hom Y Z
          H✝ : CategoryTheory.IsPushout f g h i
          H : H✝.IsVanKampen
          F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
          c' : CategoryTheory.Limits.Cocone F'
          α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
          fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
          eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
          hα : CategoryTheory.NatTrans.Equifibered α
          this : Eq (CategoryTheory.CategoryStruct.comp (F'.map CategoryTheory.Limits.Wa …
          ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cocones.precompose (CategoryTheor …
        -/
      · refine Cocones.ext (Iso.refl c'.pt) ?_
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          h : Quiver.Hom X Z
          i : Quiver.Hom Y Z
          H✝ : CategoryTheory.IsPushout f g h i
          H : H✝.IsVanKampen
          F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
          c' : CategoryTheory.Limits.Cocone F'
          α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
          fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
          eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
          hα : CategoryTheory.NatTrans.Equifibered α
          this : Eq (CategoryTheory.CategoryStruct.comp (F'.map CategoryTheory.Limits.Wa …
          ⊢ ∀ (j : CategoryTheory.Limits.WalkingSpan), Eq (CategoryTheory.CategoryStruct …
        -/
        rintro (_ | _ | _) <;> dsimp <;>
          /-
            case none
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f : Quiver.Hom W X
            g : Quiver.Hom W Y
            h : Quiver.Hom X Z
            i : Quiver.Hom Y Z
            H✝ : CategoryTheory.IsPushout f g h i
            H : H✝.IsVanKampen
            F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
            c' : CategoryTheory.Limits.Cocone F'
            α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
            fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
            eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
            hα : CategoryTheory.NatTrans.Equifibered α
            this : Eq (CategoryTheory.CategoryStruct.comp (F'.map CategoryTheory.Limits.Wa …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          simp only [c'.w, Category.assoc, Category.id_comp, Category.comp_id]
          /-
            🎉 no goals
          -/
      /-
        case mp.refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : H✝.IsVanKampen
        F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
        fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
        eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
        hα : CategoryTheory.NatTrans.Equifibered α
        ⊢ CategoryTheory.CommSq (c'.ι.app CategoryTheory.Limits.WalkingSpan.left) (α.a …
      -/
    · exact ⟨NatTrans.congr_app eα.symm _⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_3
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : H✝.IsVanKampen
        F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
        fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
        eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
        hα : CategoryTheory.NatTrans.Equifibered α
        ⊢ CategoryTheory.CommSq (c'.ι.app CategoryTheory.Limits.WalkingSpan.right) (α. …
      -/
    · exact ⟨NatTrans.congr_app eα.symm _⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_4
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : H✝.IsVanKampen
        F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
        fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
        eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
        hα : CategoryTheory.NatTrans.Equifibered α
        ⊢ CategoryTheory.CommSq (F'.map CategoryTheory.Limits.WalkingSpan.Hom.fst) (F' …
      -/
    · exact ⟨by simp⟩
      /-
        🎉 no goals
      -/
    /-
      case mp.refine_5
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      H✝ : CategoryTheory.IsPushout f g h i
      H : H✝.IsVanKampen
      F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
      fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
      eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
      hα : CategoryTheory.NatTrans.Equifibered α
      ⊢ Iff (And (CategoryTheory.IsPullback (c'.ι.app CategoryTheory.Limits.WalkingS …
    -/
    constructor
      /-
        case mp.refine_5.mp
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : H✝.IsVanKampen
        F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
        fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
        eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
        hα : CategoryTheory.NatTrans.Equifibered α
        ⊢ And (CategoryTheory.IsPullback (c'.ι.app CategoryTheory.Limits.WalkingSpan.l …
      -/
    · rintro ⟨h₁, h₂⟩ (_ | _ | _)
        /-
          case mp.refine_5.mp.intro.none
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          h : Quiver.Hom X Z
          i : Quiver.Hom Y Z
          H✝ : CategoryTheory.IsPushout f g h i
          H : H✝.IsVanKampen
          F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
          c' : CategoryTheory.Limits.Cocone F'
          α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
          fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
          eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
          hα : CategoryTheory.NatTrans.Equifibered α
          h₁ : CategoryTheory.IsPullback (c'.ι.app CategoryTheory.Limits.WalkingSpan.lef …
          h₂ : CategoryTheory.IsPullback (c'.ι.app CategoryTheory.Limits.WalkingSpan.rig …
          ⊢ CategoryTheory.IsPullback (c'.ι.app Option.none) (α.app Option.none) fα ((Ca …
        -/
      · rw [← c'.w WalkingSpan.Hom.fst]; exact (hα WalkingSpan.Hom.fst).paste_horiz h₁
                                         /-
                                           🎉 no goals
                                         -/
      /-
        case mp.refine_5.mp.intro.some.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : H✝.IsVanKampen
        F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
        fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
        eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
        hα : CategoryTheory.NatTrans.Equifibered α
        h₁ : CategoryTheory.IsPullback (c'.ι.app CategoryTheory.Limits.WalkingSpan.lef …
        h₂ : CategoryTheory.IsPullback (c'.ι.app CategoryTheory.Limits.WalkingSpan.rig …
        ⊢ CategoryTheory.IsPullback (c'.ι.app (Option.some CategoryTheory.Limits.Walki …
      -/
      exacts [h₁, h₂]
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_5.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : H✝.IsVanKampen
        F' : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (CategoryTheory.Limits.span f g)
        fα : Quiver.Hom c'.pt (CategoryTheory.Limits.PushoutCocone.mk h i ⋯).pt
        eα : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.PushoutCo …
        hα : CategoryTheory.NatTrans.Equifibered α
        ⊢ (∀ (j : CategoryTheory.Limits.WalkingSpan), CategoryTheory.IsPullback (c'.ι. …
      -/
    · intro h; exact ⟨h _, h _⟩
               /-
                 🎉 no goals
               -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      H : CategoryTheory.IsPushout f g h i
      ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
    -/
  · introv H W' hf hg hh hi w
    refine
      Iff.trans ?_ ((H w.cocone ⟨by rintro (_ | _ | _); exacts [αW, αX, αY], ?_⟩ αZ ?_ ?_).trans ?_)
    /-
      case mpr.refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      H✝ : CategoryTheory.IsPushout f g h i
      H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY g
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      ⊢ Iff (CategoryTheory.IsPushout f' g' h' i') (Nonempty (CategoryTheory.Limits. …
    -/
    rotate_left
      /-
        case mpr.refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
        W' X' Y' Z' : C
        f' : Quiver.Hom W' X'
        g' : Quiver.Hom W' Y'
        h' : Quiver.Hom X' Z'
        i' : Quiver.Hom Y' Z'
        αW : Quiver.Hom W' W
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        αZ : Quiver.Hom Z' Z
        hf : CategoryTheory.IsPullback f' αW αX f
        hg : CategoryTheory.IsPullback g' αW αY g
        hh : CategoryTheory.CommSq h' αX αZ h
        hi : CategoryTheory.CommSq i' αY αZ i
        w : CategoryTheory.CommSq f' g' h' i'
        ⊢ ∀ ⦃X_1 Y_1 : CategoryTheory.Limits.WalkingSpan⦄ (f_1 : Quiver.Hom X_1 Y_1),  …
      -/
    · rintro i _ (_ | _ | _)
        /-
          case mpr.refine_2.id
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          h : Quiver.Hom X Z
          i✝ : Quiver.Hom Y Z
          H✝ : CategoryTheory.IsPushout f g h i✝
          H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
          W' X' Y' Z' : C
          f' : Quiver.Hom W' X'
          g' : Quiver.Hom W' Y'
          h' : Quiver.Hom X' Z'
          i' : Quiver.Hom Y' Z'
          αW : Quiver.Hom W' W
          αX : Quiver.Hom X' X
          αY : Quiver.Hom Y' Y
          αZ : Quiver.Hom Z' Z
          hf : CategoryTheory.IsPullback f' αW αX f
          hg : CategoryTheory.IsPullback g' αW αY g
          hh : CategoryTheory.CommSq h' αX αZ h
          hi : CategoryTheory.CommSq i' αY αZ i✝
          w : CategoryTheory.CommSq f' g' h' i'
          i : CategoryTheory.Limits.WalkingSpan
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.span f' g').m …
        -/
      · dsimp; simp only [Functor.map_id, Category.comp_id, Category.id_comp]
               /-
                 🎉 no goals
               -/
      /-
        case mpr.refine_2.init.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
        W' X' Y' Z' : C
        f' : Quiver.Hom W' X'
        g' : Quiver.Hom W' Y'
        h' : Quiver.Hom X' Z'
        i' : Quiver.Hom Y' Z'
        αW : Quiver.Hom W' W
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        αZ : Quiver.Hom Z' Z
        hf : CategoryTheory.IsPullback f' αW αX f
        hg : CategoryTheory.IsPullback g' αW αY g
        hh : CategoryTheory.CommSq h' αX αZ h
        hi : CategoryTheory.CommSq i' αY αZ i
        w : CategoryTheory.CommSq f' g' h' i'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.span f' g').m …
      -/
      exacts [hf.w, hg.w]
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_3
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
        W' X' Y' Z' : C
        f' : Quiver.Hom W' X'
        g' : Quiver.Hom W' Y'
        h' : Quiver.Hom X' Z'
        i' : Quiver.Hom Y' Z'
        αW : Quiver.Hom W' W
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        αZ : Quiver.Hom Z' Z
        hf : CategoryTheory.IsPullback f' αW αX f
        hg : CategoryTheory.IsPullback g' αW αY g
        hh : CategoryTheory.CommSq h' αX αZ h
        hi : CategoryTheory.CommSq i' αY αZ i
        w : CategoryTheory.CommSq f' g' h' i'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun X_1 => Option.casesOn X_ …
      -/
    · ext (_ | _ | _)
        /-
          case mpr.refine_3.w.h.none
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          h : Quiver.Hom X Z
          i : Quiver.Hom Y Z
          H✝ : CategoryTheory.IsPushout f g h i
          H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
          W' X' Y' Z' : C
          f' : Quiver.Hom W' X'
          g' : Quiver.Hom W' Y'
          h' : Quiver.Hom X' Z'
          i' : Quiver.Hom Y' Z'
          αW : Quiver.Hom W' W
          αX : Quiver.Hom X' X
          αY : Quiver.Hom Y' Y
          αZ : Quiver.Hom Z' Z
          hf : CategoryTheory.IsPullback f' αW αX f
          hg : CategoryTheory.IsPullback g' αW αY g
          hh : CategoryTheory.CommSq h' αX αZ h
          hi : CategoryTheory.CommSq i' αY αZ i
          w : CategoryTheory.CommSq f' g' h' i'
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { app := fun X_1 => Option.casesOn X …
        -/
      · dsimp; rw [PushoutCocone.condition_zero]; erw [Category.assoc, hh.w, hf.w_assoc]
                                                  /-
                                                    🎉 no goals
                                                  -/
      /-
        case mpr.refine_3.w.h.some.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
        W' X' Y' Z' : C
        f' : Quiver.Hom W' X'
        g' : Quiver.Hom W' Y'
        h' : Quiver.Hom X' Z'
        i' : Quiver.Hom Y' Z'
        αW : Quiver.Hom W' W
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        αZ : Quiver.Hom Z' Z
        hf : CategoryTheory.IsPullback f' αW αX f
        hg : CategoryTheory.IsPullback g' αW αY g
        hh : CategoryTheory.CommSq h' αX αZ h
        hi : CategoryTheory.CommSq i' αY αZ i
        w : CategoryTheory.CommSq f' g' h' i'
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp { app := fun X_1 => Option.casesOn X …
      -/
      exacts [hh.w.symm, hi.w.symm]
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_4
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
        W' X' Y' Z' : C
        f' : Quiver.Hom W' X'
        g' : Quiver.Hom W' Y'
        h' : Quiver.Hom X' Z'
        i' : Quiver.Hom Y' Z'
        αW : Quiver.Hom W' W
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        αZ : Quiver.Hom Z' Z
        hf : CategoryTheory.IsPullback f' αW αX f
        hg : CategoryTheory.IsPullback g' αW αY g
        hh : CategoryTheory.CommSq h' αX αZ h
        hi : CategoryTheory.CommSq i' αY αZ i
        w : CategoryTheory.CommSq f' g' h' i'
        ⊢ CategoryTheory.NatTrans.Equifibered { app := fun X_1 => Option.casesOn X_1 α …
      -/
    · rintro i _ (_ | _ | _)
        /-
          case mpr.refine_4.id
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          h : Quiver.Hom X Z
          i✝ : Quiver.Hom Y Z
          H✝ : CategoryTheory.IsPushout f g h i✝
          H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
          W' X' Y' Z' : C
          f' : Quiver.Hom W' X'
          g' : Quiver.Hom W' Y'
          h' : Quiver.Hom X' Z'
          i' : Quiver.Hom Y' Z'
          αW : Quiver.Hom W' W
          αX : Quiver.Hom X' X
          αY : Quiver.Hom Y' Y
          αZ : Quiver.Hom Z' Z
          hf : CategoryTheory.IsPullback f' αW αX f
          hg : CategoryTheory.IsPullback g' αW αY g
          hh : CategoryTheory.CommSq h' αX αZ h
          hi : CategoryTheory.CommSq i' αY αZ i✝
          w : CategoryTheory.CommSq f' g' h' i'
          i : CategoryTheory.Limits.WalkingSpan
          ⊢ CategoryTheory.IsPullback ((CategoryTheory.Limits.span f' g').map (CategoryT …
        -/
      · dsimp; simp_rw [Functor.map_id]
        /-
          case mpr.refine_4.id
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          h : Quiver.Hom X Z
          i✝ : Quiver.Hom Y Z
          H✝ : CategoryTheory.IsPushout f g h i✝
          H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
          W' X' Y' Z' : C
          f' : Quiver.Hom W' X'
          g' : Quiver.Hom W' Y'
          h' : Quiver.Hom X' Z'
          i' : Quiver.Hom Y' Z'
          αW : Quiver.Hom W' W
          αX : Quiver.Hom X' X
          αY : Quiver.Hom Y' Y
          αZ : Quiver.Hom Z' Z
          hf : CategoryTheory.IsPullback f' αW αX f
          hg : CategoryTheory.IsPullback g' αW αY g
          hh : CategoryTheory.CommSq h' αX αZ h
          hi : CategoryTheory.CommSq i' αY αZ i✝
          w : CategoryTheory.CommSq f' g' h' i'
          i : CategoryTheory.Limits.WalkingSpan
          ⊢ CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.id ((CategoryTheory …
        -/
        exact IsPullback.of_horiz_isIso ⟨by rw [Category.comp_id, Category.id_comp]⟩
        /-
          🎉 no goals
        -/
      /-
        case mpr.refine_4.init.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
        W' X' Y' Z' : C
        f' : Quiver.Hom W' X'
        g' : Quiver.Hom W' Y'
        h' : Quiver.Hom X' Z'
        i' : Quiver.Hom Y' Z'
        αW : Quiver.Hom W' W
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        αZ : Quiver.Hom Z' Z
        hf : CategoryTheory.IsPullback f' αW αX f
        hg : CategoryTheory.IsPullback g' αW αY g
        hh : CategoryTheory.CommSq h' αX αZ h
        hi : CategoryTheory.CommSq i' αY αZ i
        w : CategoryTheory.CommSq f' g' h' i'
        ⊢ CategoryTheory.IsPullback ((CategoryTheory.Limits.span f' g').map (CategoryT …
      -/
      exacts [hf, hg]
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_5
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
        W' X' Y' Z' : C
        f' : Quiver.Hom W' X'
        g' : Quiver.Hom W' Y'
        h' : Quiver.Hom X' Z'
        i' : Quiver.Hom Y' Z'
        αW : Quiver.Hom W' W
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        αZ : Quiver.Hom Z' Z
        hf : CategoryTheory.IsPullback f' αW αX f
        hg : CategoryTheory.IsPullback g' αW αY g
        hh : CategoryTheory.CommSq h' αX αZ h
        hi : CategoryTheory.CommSq i' αY αZ i
        w : CategoryTheory.CommSq f' g' h' i'
        ⊢ Iff (∀ (j : CategoryTheory.Limits.WalkingSpan), CategoryTheory.IsPullback (w …
      -/
    · constructor
        /-
          case mpr.refine_5.mp
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          h : Quiver.Hom X Z
          i : Quiver.Hom Y Z
          H✝ : CategoryTheory.IsPushout f g h i
          H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
          W' X' Y' Z' : C
          f' : Quiver.Hom W' X'
          g' : Quiver.Hom W' Y'
          h' : Quiver.Hom X' Z'
          i' : Quiver.Hom Y' Z'
          αW : Quiver.Hom W' W
          αX : Quiver.Hom X' X
          αY : Quiver.Hom Y' Y
          αZ : Quiver.Hom Z' Z
          hf : CategoryTheory.IsPullback f' αW αX f
          hg : CategoryTheory.IsPullback g' αW αY g
          hh : CategoryTheory.CommSq h' αX αZ h
          hi : CategoryTheory.CommSq i' αY αZ i
          w : CategoryTheory.CommSq f' g' h' i'
          ⊢ (∀ (j : CategoryTheory.Limits.WalkingSpan), CategoryTheory.IsPullback (w.coc …
        -/
      · intro h; exact ⟨h WalkingCospan.left, h WalkingCospan.right⟩
                 /-
                   🎉 no goals
                 -/
        /-
          case mpr.refine_5.mpr
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          h : Quiver.Hom X Z
          i : Quiver.Hom Y Z
          H✝ : CategoryTheory.IsPushout f g h i
          H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
          W' X' Y' Z' : C
          f' : Quiver.Hom W' X'
          g' : Quiver.Hom W' Y'
          h' : Quiver.Hom X' Z'
          i' : Quiver.Hom Y' Z'
          αW : Quiver.Hom W' W
          αX : Quiver.Hom X' X
          αY : Quiver.Hom Y' Y
          αZ : Quiver.Hom Z' Z
          hf : CategoryTheory.IsPullback f' αW αX f
          hg : CategoryTheory.IsPullback g' αW αY g
          hh : CategoryTheory.CommSq h' αX αZ h
          hi : CategoryTheory.CommSq i' αY αZ i
          w : CategoryTheory.CommSq f' g' h' i'
          ⊢ And (CategoryTheory.IsPullback h' αX αZ h) (CategoryTheory.IsPullback i' αY  …
        -/
      · rintro ⟨h₁, h₂⟩ (_ | _ | _)
          /-
            case mpr.refine_5.mpr.intro.none
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f : Quiver.Hom W X
            g : Quiver.Hom W Y
            h : Quiver.Hom X Z
            i : Quiver.Hom Y Z
            H✝ : CategoryTheory.IsPushout f g h i
            H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
            W' X' Y' Z' : C
            f' : Quiver.Hom W' X'
            g' : Quiver.Hom W' Y'
            h' : Quiver.Hom X' Z'
            i' : Quiver.Hom Y' Z'
            αW : Quiver.Hom W' W
            αX : Quiver.Hom X' X
            αY : Quiver.Hom Y' Y
            αZ : Quiver.Hom Z' Z
            hf : CategoryTheory.IsPullback f' αW αX f
            hg : CategoryTheory.IsPullback g' αW αY g
            hh : CategoryTheory.CommSq h' αX αZ h
            hi : CategoryTheory.CommSq i' αY αZ i
            w : CategoryTheory.CommSq f' g' h' i'
            h₁ : CategoryTheory.IsPullback h' αX αZ h
            h₂ : CategoryTheory.IsPullback i' αY αZ i
            ⊢ CategoryTheory.IsPullback (w.cocone.ι.app Option.none) ({ app := fun X_1 =>  …
          -/
        · dsimp; rw [PushoutCocone.condition_zero]; exact hf.paste_horiz h₁
                                                    /-
                                                      🎉 no goals
                                                    -/
        /-
          case mpr.refine_5.mpr.intro.some.left
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom W X
          g : Quiver.Hom W Y
          h : Quiver.Hom X Z
          i : Quiver.Hom Y Z
          H✝ : CategoryTheory.IsPushout f g h i
          H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
          W' X' Y' Z' : C
          f' : Quiver.Hom W' X'
          g' : Quiver.Hom W' Y'
          h' : Quiver.Hom X' Z'
          i' : Quiver.Hom Y' Z'
          αW : Quiver.Hom W' W
          αX : Quiver.Hom X' X
          αY : Quiver.Hom Y' Y
          αZ : Quiver.Hom Z' Z
          hf : CategoryTheory.IsPullback f' αW αX f
          hg : CategoryTheory.IsPullback g' αW αY g
          hh : CategoryTheory.CommSq h' αX αZ h
          hi : CategoryTheory.CommSq i' αY αZ i
          w : CategoryTheory.CommSq f' g' h' i'
          h₁ : CategoryTheory.IsPullback h' αX αZ h
          h₂ : CategoryTheory.IsPullback i' αY αZ i
          ⊢ CategoryTheory.IsPullback (w.cocone.ι.app (Option.some CategoryTheory.Limits …
        -/
        exacts [h₁, h₂]
        /-
          🎉 no goals
        -/
      /-
        case mpr.refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        H✝ : CategoryTheory.IsPushout f g h i
        H : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk  …
        W' X' Y' Z' : C
        f' : Quiver.Hom W' X'
        g' : Quiver.Hom W' Y'
        h' : Quiver.Hom X' Z'
        i' : Quiver.Hom Y' Z'
        αW : Quiver.Hom W' W
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        αZ : Quiver.Hom Z' Z
        hf : CategoryTheory.IsPullback f' αW αX f
        hg : CategoryTheory.IsPullback g' αW αY g
        hh : CategoryTheory.CommSq h' αX αZ h
        hi : CategoryTheory.CommSq i' αY αZ i
        w : CategoryTheory.CommSq f' g' h' i'
        ⊢ Iff (CategoryTheory.IsPushout f' g' h' i') (Nonempty (CategoryTheory.Limits. …
      -/
    · exact ⟨fun h => h.2, fun h => ⟨w, h⟩⟩
      /-
        🎉 no goals
      -/


theorem is_coprod_iff_isPushout {X E Y YE : C} (c : BinaryCofan X E) (hc : IsColimit c) {f : X ⟶ Y}
    {iY : Y ⟶ YE} {fE : c.pt ⟶ YE} (H : CommSq f c.inl iY fE) :
    Nonempty (IsColimit (BinaryCofan.mk (c.inr ≫ fE) iY)) ↔ IsPushout f c.inl iY fE := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X E Y YE : C
    c : CategoryTheory.Limits.BinaryCofan X E
    hc : CategoryTheory.Limits.IsColimit c
    f : Quiver.Hom X Y
    iY : Quiver.Hom Y YE
    fE : Quiver.Hom c.pt YE
    H : CategoryTheory.CommSq f c.inl iY fE
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Binary …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X E Y YE : C
      c : CategoryTheory.Limits.BinaryCofan X E
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom X Y
      iY : Quiver.Hom Y YE
      fE : Quiver.Hom c.pt YE
      H : CategoryTheory.CommSq f c.inl iY fE
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan …
    -/
  · rintro ⟨h⟩
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X E Y YE : C
      c : CategoryTheory.Limits.BinaryCofan X E
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom X Y
      iY : Quiver.Hom Y YE
      fE : Quiver.Hom c.pt YE
      H : CategoryTheory.CommSq f c.inl iY fE
      h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
      ⊢ CategoryTheory.IsPushout f c.inl iY fE
    -/
    refine ⟨H, ⟨Limits.PushoutCocone.isColimitAux' _ ?_⟩⟩
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X E Y YE : C
      c : CategoryTheory.Limits.BinaryCofan X E
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom X Y
      iY : Quiver.Hom Y YE
      fE : Quiver.Hom c.pt YE
      H : CategoryTheory.CommSq f c.inl iY fE
      h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
      ⊢ (s : CategoryTheory.Limits.PushoutCocone f c.inl) → Subtype fun l => And (Eq …
    -/
    intro s
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X E Y YE : C
      c : CategoryTheory.Limits.BinaryCofan X E
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom X Y
      iY : Quiver.Hom Y YE
      fE : Quiver.Hom c.pt YE
      H : CategoryTheory.CommSq f c.inl iY fE
      h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
      s : CategoryTheory.Limits.PushoutCocone f c.inl
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory …
    -/
    dsimp only [PushoutCocone.inr, PushoutCocone.mk] -- Porting note: Originally `dsimp`
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X E Y YE : C
      c : CategoryTheory.Limits.BinaryCofan X E
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom X Y
      iY : Quiver.Hom Y YE
      fE : Quiver.Hom c.pt YE
      H : CategoryTheory.CommSq f c.inl iY fE
      h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
      s : CategoryTheory.Limits.PushoutCocone f c.inl
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory …
    -/
    refine ⟨h.desc (BinaryCofan.mk (c.inr ≫ s.inr) s.inl), h.fac _ ⟨WalkingPair.right⟩, ?_, ?_⟩
      /-
        case mp.intro.refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X E Y YE : C
        c : CategoryTheory.Limits.BinaryCofan X E
        hc : CategoryTheory.Limits.IsColimit c
        f : Quiver.Hom X Y
        iY : Quiver.Hom Y YE
        fE : Quiver.Hom c.pt YE
        H : CategoryTheory.CommSq f c.inl iY fE
        h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
        s : CategoryTheory.Limits.PushoutCocone f c.inl
        ⊢ Eq (CategoryTheory.CategoryStruct.comp fE (h.desc (CategoryTheory.Limits.Bin …
      -/
    · apply BinaryCofan.IsColimit.hom_ext hc
        /-
          case mp.intro.refine_1.h₁
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X E Y YE : C
          c : CategoryTheory.Limits.BinaryCofan X E
          hc : CategoryTheory.Limits.IsColimit c
          f : Quiver.Hom X Y
          iY : Quiver.Hom Y YE
          fE : Quiver.Hom c.pt YE
          H : CategoryTheory.CommSq f c.inl iY fE
          h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
          s : CategoryTheory.Limits.PushoutCocone f c.inl
          ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inl (CategoryTheory.CategoryStruct. …
        -/
      · rw [← H.w_assoc]; erw [h.fac _ ⟨WalkingPair.right⟩]; exact s.condition
                                                             /-
                                                               🎉 no goals
                                                             -/
        /-
          case mp.intro.refine_1.h₂
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X E Y YE : C
          c : CategoryTheory.Limits.BinaryCofan X E
          hc : CategoryTheory.Limits.IsColimit c
          f : Quiver.Hom X Y
          iY : Quiver.Hom Y YE
          fE : Quiver.Hom c.pt YE
          H : CategoryTheory.CommSq f c.inl iY fE
          h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
          s : CategoryTheory.Limits.PushoutCocone f c.inl
          ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inr (CategoryTheory.CategoryStruct. …
        -/
      · rw [← Category.assoc]; exact h.fac _ ⟨WalkingPair.left⟩
                               /-
                                 🎉 no goals
                               -/
      /-
        case mp.intro.refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X E Y YE : C
        c : CategoryTheory.Limits.BinaryCofan X E
        hc : CategoryTheory.Limits.IsColimit c
        f : Quiver.Hom X Y
        iY : Quiver.Hom Y YE
        fE : Quiver.Hom c.pt YE
        H : CategoryTheory.CommSq f c.inl iY fE
        h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
        s : CategoryTheory.Limits.PushoutCocone f c.inl
        ⊢ ∀ {m : Quiver.Hom YE s.pt}, Eq (CategoryTheory.CategoryStruct.comp (Category …
      -/
    · intro m e₁ e₂
      /-
        case mp.intro.refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X E Y YE : C
        c : CategoryTheory.Limits.BinaryCofan X E
        hc : CategoryTheory.Limits.IsColimit c
        f : Quiver.Hom X Y
        iY : Quiver.Hom Y YE
        fE : Quiver.Hom c.pt YE
        H : CategoryTheory.CommSq f c.inl iY fE
        h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
        s : CategoryTheory.Limits.PushoutCocone f c.inl
        m : Quiver.Hom YE s.pt
        e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp fE m) (s.ι.app CategoryTheory.Limi …
        ⊢ Eq m (h.desc (CategoryTheory.Limits.BinaryCofan.mk (CategoryTheory.CategoryS …
      -/
      apply BinaryCofan.IsColimit.hom_ext h
        /-
          case mp.intro.refine_2.h₁
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X E Y YE : C
          c : CategoryTheory.Limits.BinaryCofan X E
          hc : CategoryTheory.Limits.IsColimit c
          f : Quiver.Hom X Y
          iY : Quiver.Hom Y YE
          fE : Quiver.Hom c.pt YE
          H : CategoryTheory.CommSq f c.inl iY fE
          h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
          s : CategoryTheory.Limits.PushoutCocone f c.inl
          m : Quiver.Hom YE s.pt
          e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
          e₂ : Eq (CategoryTheory.CategoryStruct.comp fE m) (s.ι.app CategoryTheory.Limi …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan.mk …
        -/
      · dsimp only [BinaryCofan.mk, id] -- Porting note: Originally `dsimp`
        /-
          case mp.intro.refine_2.h₁
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X E Y YE : C
          c : CategoryTheory.Limits.BinaryCofan X E
          hc : CategoryTheory.Limits.IsColimit c
          f : Quiver.Hom X Y
          iY : Quiver.Hom Y YE
          fE : Quiver.Hom c.pt YE
          H : CategoryTheory.CommSq f c.inl iY fE
          h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
          s : CategoryTheory.Limits.PushoutCocone f c.inl
          m : Quiver.Hom YE s.pt
          e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
          e₂ : Eq (CategoryTheory.CategoryStruct.comp fE m) (s.ι.app CategoryTheory.Limi …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan.in …
        -/
        rw [Category.assoc, e₂, eq_comm]; exact h.fac _ ⟨WalkingPair.left⟩
                                          /-
                                            🎉 no goals
                                          -/
        /-
          case mp.intro.refine_2.h₂
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X E Y YE : C
          c : CategoryTheory.Limits.BinaryCofan X E
          hc : CategoryTheory.Limits.IsColimit c
          f : Quiver.Hom X Y
          iY : Quiver.Hom Y YE
          fE : Quiver.Hom c.pt YE
          H : CategoryTheory.CommSq f c.inl iY fE
          h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Cat …
          s : CategoryTheory.Limits.PushoutCocone f c.inl
          m : Quiver.Hom YE s.pt
          e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
          e₂ : Eq (CategoryTheory.CategoryStruct.comp fE m) (s.ι.app CategoryTheory.Limi …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan.mk …
        -/
      · refine e₁.trans (Eq.symm ?_); exact h.fac _ _
                                      /-
                                        🎉 no goals
                                      -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X E Y YE : C
      c : CategoryTheory.Limits.BinaryCofan X E
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom X Y
      iY : Quiver.Hom Y YE
      fE : Quiver.Hom c.pt YE
      H : CategoryTheory.CommSq f c.inl iY fE
      ⊢ CategoryTheory.IsPushout f c.inl iY fE → Nonempty (CategoryTheory.Limits.IsC …
    -/
  · refine fun H => ⟨?_⟩
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X E Y YE : C
      c : CategoryTheory.Limits.BinaryCofan X E
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom X Y
      iY : Quiver.Hom Y YE
      fE : Quiver.Hom c.pt YE
      H✝ : CategoryTheory.CommSq f c.inl iY fE
      H : CategoryTheory.IsPushout f c.inl iY fE
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Categ …
    -/
    fapply Limits.BinaryCofan.isColimitMk
    · exact fun s => H.isColimit.desc (PushoutCocone.mk s.inr _ <|
        (hc.fac (BinaryCofan.mk (f ≫ s.inr) s.inl) ⟨WalkingPair.left⟩).symm)
      /-
        case mpr.fac_left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X E Y YE : C
        c : CategoryTheory.Limits.BinaryCofan X E
        hc : CategoryTheory.Limits.IsColimit c
        f : Quiver.Hom X Y
        iY : Quiver.Hom Y YE
        fE : Quiver.Hom c.pt YE
        H✝ : CategoryTheory.CommSq f c.inl iY fE
        H : CategoryTheory.IsPushout f c.inl iY fE
        ⊢ ∀ (s : CategoryTheory.Limits.BinaryCofan ((CategoryTheory.Limits.pair X E).o …
      -/
    · intro s
      /-
        case mpr.fac_left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X E Y YE : C
        c : CategoryTheory.Limits.BinaryCofan X E
        hc : CategoryTheory.Limits.IsColimit c
        f : Quiver.Hom X Y
        iY : Quiver.Hom Y YE
        fE : Quiver.Hom c.pt YE
        H✝ : CategoryTheory.CommSq f c.inl iY fE
        H : CategoryTheory.IsPushout f c.inl iY fE
        s : CategoryTheory.Limits.BinaryCofan ((CategoryTheory.Limits.pair X E).obj {  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp c …
      -/
      erw [Category.assoc, H.isColimit.fac _ WalkingSpan.right, hc.fac]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
      /-
        case mpr.fac_right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X E Y YE : C
        c : CategoryTheory.Limits.BinaryCofan X E
        hc : CategoryTheory.Limits.IsColimit c
        f : Quiver.Hom X Y
        iY : Quiver.Hom Y YE
        fE : Quiver.Hom c.pt YE
        H✝ : CategoryTheory.CommSq f c.inl iY fE
        H : CategoryTheory.IsPushout f c.inl iY fE
        ⊢ ∀ (s : CategoryTheory.Limits.BinaryCofan ((CategoryTheory.Limits.pair X E).o …
      -/
    · intro s; exact H.isColimit.fac _ WalkingSpan.left
               /-
                 🎉 no goals
               -/
      /-
        case mpr.uniq
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X E Y YE : C
        c : CategoryTheory.Limits.BinaryCofan X E
        hc : CategoryTheory.Limits.IsColimit c
        f : Quiver.Hom X Y
        iY : Quiver.Hom Y YE
        fE : Quiver.Hom c.pt YE
        H✝ : CategoryTheory.CommSq f c.inl iY fE
        H : CategoryTheory.IsPushout f c.inl iY fE
        ⊢ ∀ (s : CategoryTheory.Limits.BinaryCofan ((CategoryTheory.Limits.pair X E).o …
      -/
    · intro s m e₁ e₂
      /-
        case mpr.uniq
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X E Y YE : C
        c : CategoryTheory.Limits.BinaryCofan X E
        hc : CategoryTheory.Limits.IsColimit c
        f : Quiver.Hom X Y
        iY : Quiver.Hom Y YE
        fE : Quiver.Hom c.pt YE
        H✝ : CategoryTheory.CommSq f c.inl iY fE
        H : CategoryTheory.IsPushout f c.inl iY fE
        s : CategoryTheory.Limits.BinaryCofan ((CategoryTheory.Limits.pair X E).obj {  …
        m : Quiver.Hom YE s.pt
        e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp iY m) s.inr
        ⊢ Eq m (H.isColimit.desc (CategoryTheory.Limits.PushoutCocone.mk s.inr (hc.des …
      -/
      apply PushoutCocone.IsColimit.hom_ext H.isColimit
        /-
          case mpr.uniq.h₀
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X E Y YE : C
          c : CategoryTheory.Limits.BinaryCofan X E
          hc : CategoryTheory.Limits.IsColimit c
          f : Quiver.Hom X Y
          iY : Quiver.Hom Y YE
          fE : Quiver.Hom c.pt YE
          H✝ : CategoryTheory.CommSq f c.inl iY fE
          H : CategoryTheory.IsPushout f c.inl iY fE
          s : CategoryTheory.Limits.BinaryCofan ((CategoryTheory.Limits.pair X E).obj {  …
          m : Quiver.Hom YE s.pt
          e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
          e₂ : Eq (CategoryTheory.CategoryStruct.comp iY m) s.inr
          ⊢ Eq (CategoryTheory.CategoryStruct.comp H.cocone.inl m) (CategoryTheory.Categ …
        -/
      · symm; exact (H.isColimit.fac _ WalkingSpan.left).trans e₂.symm
              /-
                🎉 no goals
              -/
        /-
          case mpr.uniq.h₁
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X E Y YE : C
          c : CategoryTheory.Limits.BinaryCofan X E
          hc : CategoryTheory.Limits.IsColimit c
          f : Quiver.Hom X Y
          iY : Quiver.Hom Y YE
          fE : Quiver.Hom c.pt YE
          H✝ : CategoryTheory.CommSq f c.inl iY fE
          H : CategoryTheory.IsPushout f c.inl iY fE
          s : CategoryTheory.Limits.BinaryCofan ((CategoryTheory.Limits.pair X E).obj {  …
          m : Quiver.Hom YE s.pt
          e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
          e₂ : Eq (CategoryTheory.CategoryStruct.comp iY m) s.inr
          ⊢ Eq (CategoryTheory.CategoryStruct.comp H.cocone.inr m) (CategoryTheory.Categ …
        -/
      · rw [H.isColimit.fac _ WalkingSpan.right]
        /-
          case mpr.uniq.h₁
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X E Y YE : C
          c : CategoryTheory.Limits.BinaryCofan X E
          hc : CategoryTheory.Limits.IsColimit c
          f : Quiver.Hom X Y
          iY : Quiver.Hom Y YE
          fE : Quiver.Hom c.pt YE
          H✝ : CategoryTheory.CommSq f c.inl iY fE
          H : CategoryTheory.IsPushout f c.inl iY fE
          s : CategoryTheory.Limits.BinaryCofan ((CategoryTheory.Limits.pair X E).obj {  …
          m : Quiver.Hom YE s.pt
          e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
          e₂ : Eq (CategoryTheory.CategoryStruct.comp iY m) s.inr
          ⊢ Eq (CategoryTheory.CategoryStruct.comp H.cocone.inr m) ((CategoryTheory.Limi …
        -/
        apply BinaryCofan.IsColimit.hom_ext hc
          /-
            case mpr.uniq.h₁.h₁
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X E Y YE : C
            c : CategoryTheory.Limits.BinaryCofan X E
            hc : CategoryTheory.Limits.IsColimit c
            f : Quiver.Hom X Y
            iY : Quiver.Hom Y YE
            fE : Quiver.Hom c.pt YE
            H✝ : CategoryTheory.CommSq f c.inl iY fE
            H : CategoryTheory.IsPushout f c.inl iY fE
            s : CategoryTheory.Limits.BinaryCofan ((CategoryTheory.Limits.pair X E).obj {  …
            m : Quiver.Hom YE s.pt
            e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
            e₂ : Eq (CategoryTheory.CategoryStruct.comp iY m) s.inr
            ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inl (CategoryTheory.CategoryStruct. …
          -/
        · erw [hc.fac, ← H.w_assoc, e₂]; rfl
                                         /-
                                           🎉 no goals
                                         -/
          /-
            case mpr.uniq.h₁.h₂
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X E Y YE : C
            c : CategoryTheory.Limits.BinaryCofan X E
            hc : CategoryTheory.Limits.IsColimit c
            f : Quiver.Hom X Y
            iY : Quiver.Hom Y YE
            fE : Quiver.Hom c.pt YE
            H✝ : CategoryTheory.CommSq f c.inl iY fE
            H : CategoryTheory.IsPushout f c.inl iY fE
            s : CategoryTheory.Limits.BinaryCofan ((CategoryTheory.Limits.pair X E).obj {  …
            m : Quiver.Hom YE s.pt
            e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
            e₂ : Eq (CategoryTheory.CategoryStruct.comp iY m) s.inr
            ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inr (CategoryTheory.CategoryStruct. …
          -/
        · refine ((Category.assoc _ _ _).symm.trans e₁).trans ?_; symm; exact hc.fac _ _
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem IsPushout.isVanKampen_inl {W E X Z : C} (c : BinaryCofan W E) [FinitaryExtensive C]
    [HasPullbacks C] (hc : IsColimit c) (f : W ⟶ X) (h : X ⟶ Z) (i : c.pt ⟶ Z)
    (H : IsPushout f c.inl h i) : H.IsVanKampen := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W E X Z : C
    c : CategoryTheory.Limits.BinaryCofan W E
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    hc : CategoryTheory.Limits.IsColimit c
    f : Quiver.Hom W X
    h : Quiver.Hom X Z
    i : Quiver.Hom c.pt Z
    H : CategoryTheory.IsPushout f c.inl h i
    ⊢ H.IsVanKampen
  -/
  obtain ⟨hc₁⟩ := (is_coprod_iff_isPushout c hc H.1).mpr H
  /-
    case intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W E X Z : C
    c : CategoryTheory.Limits.BinaryCofan W E
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    hc : CategoryTheory.Limits.IsColimit c
    f : Quiver.Hom W X
    h : Quiver.Hom X Z
    i : Quiver.Hom c.pt Z
    H : CategoryTheory.IsPushout f c.inl h i
    hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
    ⊢ H.IsVanKampen
  -/
  introv W' hf hg hh hi w
  obtain ⟨hc₂⟩ := ((BinaryCofan.isVanKampen_iff _).mp (FinitaryExtensive.vanKampen c hc)
    (BinaryCofan.mk _ (pullback.fst _ _)) _ _ _ hg.w.symm pullback.condition.symm).mpr
    ⟨hg, IsPullback.of_hasPullback αY c.inr⟩
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W E X Z : C
    c : CategoryTheory.Limits.BinaryCofan W E
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    hc : CategoryTheory.Limits.IsColimit c
    f : Quiver.Hom W X
    h : Quiver.Hom X Z
    i : Quiver.Hom c.pt Z
    H : CategoryTheory.IsPushout f c.inl h i
    hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
    W' X' Y' Z' : C
    f' : Quiver.Hom W' X'
    g' : Quiver.Hom W' Y'
    h' : Quiver.Hom X' Z'
    i' : Quiver.Hom Y' Z'
    αW : Quiver.Hom W' W
    αX : Quiver.Hom X' X
    αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
    αZ : Quiver.Hom Z' Z
    hf : CategoryTheory.IsPullback f' αW αX f
    hg : CategoryTheory.IsPullback g' αW αY c.inl
    hh : CategoryTheory.CommSq h' αX αZ h
    hi : CategoryTheory.CommSq i' αY αZ i
    w : CategoryTheory.CommSq f' g' h' i'
    hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
    ⊢ Iff (CategoryTheory.IsPushout f' g' h' i') (And (CategoryTheory.IsPullback h …
  -/
  refine (is_coprod_iff_isPushout _ hc₂ w).symm.trans ?_
  refine ((BinaryCofan.isVanKampen_iff _).mp (FinitaryExtensive.vanKampen _ hc₁)
    (BinaryCofan.mk _ _) (pullback.snd _ _) _ _ ?_ hh.w.symm).trans ?_
    /-
      case intro.intro.refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W E X Z : C
      c : CategoryTheory.Limits.BinaryCofan W E
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom W X
      h : Quiver.Hom X Z
      i : Quiver.Hom c.pt Z
      H : CategoryTheory.IsPushout f c.inl h i
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY c.inl
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd α …
    -/
  · dsimp; rw [← pullback.condition_assoc, Category.assoc, hi.w]
           /-
             🎉 no goals
           -/
  /-
    case intro.intro.refine_2
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W E X Z : C
    c : CategoryTheory.Limits.BinaryCofan W E
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    hc : CategoryTheory.Limits.IsColimit c
    f : Quiver.Hom W X
    h : Quiver.Hom X Z
    i : Quiver.Hom c.pt Z
    H : CategoryTheory.IsPushout f c.inl h i
    hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
    W' X' Y' Z' : C
    f' : Quiver.Hom W' X'
    g' : Quiver.Hom W' Y'
    h' : Quiver.Hom X' Z'
    i' : Quiver.Hom Y' Z'
    αW : Quiver.Hom W' W
    αX : Quiver.Hom X' X
    αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
    αZ : Quiver.Hom Z' Z
    hf : CategoryTheory.IsPullback f' αW αX f
    hg : CategoryTheory.IsPullback g' αW αY c.inl
    hh : CategoryTheory.CommSq h' αX αZ h
    hi : CategoryTheory.CommSq i' αY αZ i
    w : CategoryTheory.CommSq f' g' h' i'
    hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
    ⊢ Iff (And (CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (C …
  -/
  constructor
    /-
      case intro.intro.refine_2.mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W E X Z : C
      c : CategoryTheory.Limits.BinaryCofan W E
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom W X
      h : Quiver.Hom X Z
      i : Quiver.Hom c.pt Z
      H : CategoryTheory.IsPushout f c.inl h i
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY c.inl
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
      ⊢ And (CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Catego …
    -/
  · rintro ⟨hc₃, hc₄⟩
    /-
      case intro.intro.refine_2.mp.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W E X Z : C
      c : CategoryTheory.Limits.BinaryCofan W E
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom W X
      h : Quiver.Hom X Z
      i : Quiver.Hom c.pt Z
      H : CategoryTheory.IsPushout f c.inl h i
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY c.inl
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
      hc₃ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Categor …
      hc₄ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Categor …
      ⊢ And (CategoryTheory.IsPullback h' αX αZ h) (CategoryTheory.IsPullback i' αY  …
    -/
    refine ⟨hc₄, ?_⟩
    /-
      case intro.intro.refine_2.mp.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W E X Z : C
      c : CategoryTheory.Limits.BinaryCofan W E
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom W X
      h : Quiver.Hom X Z
      i : Quiver.Hom c.pt Z
      H : CategoryTheory.IsPushout f c.inl h i
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY c.inl
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
      hc₃ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Categor …
      hc₄ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Categor …
      ⊢ CategoryTheory.IsPullback i' αY αZ i
    -/
    let Y'' := pullback αZ i
    /-
      case intro.intro.refine_2.mp.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W E X Z : C
      c : CategoryTheory.Limits.BinaryCofan W E
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom W X
      h : Quiver.Hom X Z
      i : Quiver.Hom c.pt Z
      H : CategoryTheory.IsPushout f c.inl h i
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY c.inl
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
      hc₃ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Categor …
      hc₄ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Categor …
      Y'' : C := CategoryTheory.Limits.pullback αZ i
      ⊢ CategoryTheory.IsPullback i' αY αZ i
    -/
    let cmp : Y' ⟶ Y'' := pullback.lift i' αY hi.w
    have e₁ : (g' ≫ cmp) ≫ pullback.snd _ _ = αW ≫ c.inl := by
      rw [Category.assoc, pullback.lift_snd, hg.w]
    have e₂ : (pullback.fst _ _ ≫ cmp : pullback αY c.inr ⟶ _) ≫ pullback.snd _ _ =
        pullback.snd _ _ ≫ c.inr := by
      rw [Category.assoc, pullback.lift_snd, pullback.condition]
    obtain ⟨hc₄⟩ := ((BinaryCofan.isVanKampen_iff _).mp (FinitaryExtensive.vanKampen c hc)
      (BinaryCofan.mk _ _) αW _ _ e₁.symm e₂.symm).mpr <| by
        constructor
        · apply IsPullback.of_right _ e₁ (IsPullback.of_hasPullback _ _)
          rw [Category.assoc, pullback.lift_fst, ← H.w, ← w.w]; exact hf.paste_horiz hc₄
        · apply IsPullback.of_right _ e₂ (IsPullback.of_hasPullback _ _)
          rw [Category.assoc, pullback.lift_fst]; exact hc₃
    /-
      case intro.intro.refine_2.mp.intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W E X Z : C
      c : CategoryTheory.Limits.BinaryCofan W E
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom W X
      h : Quiver.Hom X Z
      i : Quiver.Hom c.pt Z
      H : CategoryTheory.IsPushout f c.inl h i
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY c.inl
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
      hc₃ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Categor …
      hc₄✝ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Catego …
      Y'' : C := CategoryTheory.Limits.pullback αZ i
      cmp : Quiver.Hom Y' Y'' := CategoryTheory.Limits.pullback.lift i' αY ⋯
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      hc₄ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      ⊢ CategoryTheory.IsPullback i' αY αZ i
    -/
    rw [← Category.id_comp αZ, ← show cmp ≫ pullback.snd _ _ = αY from pullback.lift_snd _ _ _]
    /-
      case intro.intro.refine_2.mp.intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W E X Z : C
      c : CategoryTheory.Limits.BinaryCofan W E
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom W X
      h : Quiver.Hom X Z
      i : Quiver.Hom c.pt Z
      H : CategoryTheory.IsPushout f c.inl h i
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY c.inl
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
      hc₃ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Categor …
      hc₄✝ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Catego …
      Y'' : C := CategoryTheory.Limits.pullback αZ i
      cmp : Quiver.Hom Y' Y'' := CategoryTheory.Limits.pullback.lift i' αY ⋯
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      hc₄ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      ⊢ CategoryTheory.IsPullback i' (CategoryTheory.CategoryStruct.comp cmp (Catego …
    -/
    apply IsPullback.paste_vert _ (IsPullback.of_hasPullback αZ i)
    have : cmp = (hc₂.coconePointUniqueUpToIso hc₄).hom := by
      apply BinaryCofan.IsColimit.hom_ext hc₂
      exacts [(hc₂.comp_coconePointUniqueUpToIso_hom hc₄ ⟨WalkingPair.left⟩).symm,
        (hc₂.comp_coconePointUniqueUpToIso_hom hc₄ ⟨WalkingPair.right⟩).symm]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W E X Z : C
      c : CategoryTheory.Limits.BinaryCofan W E
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom W X
      h : Quiver.Hom X Z
      i : Quiver.Hom c.pt Z
      H : CategoryTheory.IsPushout f c.inl h i
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY c.inl
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
      hc₃ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Categor …
      hc₄✝ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Catego …
      Y'' : C := CategoryTheory.Limits.pullback αZ i
      cmp : Quiver.Hom Y' Y'' := CategoryTheory.Limits.pullback.lift i' αY ⋯
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      hc₄ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      this : Eq cmp (hc₂.coconePointUniqueUpToIso hc₄).hom
      ⊢ CategoryTheory.IsPullback i' cmp (CategoryTheory.CategoryStruct.id Z') (Cate …
    -/
    rw [this]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W E X Z : C
      c : CategoryTheory.Limits.BinaryCofan W E
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom W X
      h : Quiver.Hom X Z
      i : Quiver.Hom c.pt Z
      H : CategoryTheory.IsPushout f c.inl h i
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY c.inl
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
      hc₃ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Categor …
      hc₄✝ : CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Catego …
      Y'' : C := CategoryTheory.Limits.pullback αZ i
      cmp : Quiver.Hom Y' Y'' := CategoryTheory.Limits.pullback.lift i' αY ⋯
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      hc₄ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      this : Eq cmp (hc₂.coconePointUniqueUpToIso hc₄).hom
      ⊢ CategoryTheory.IsPullback i' (hc₂.coconePointUniqueUpToIso hc₄).hom (Categor …
    -/
    exact IsPullback.of_vert_isIso ⟨by rw [← this, Category.comp_id, pullback.lift_fst]⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2.mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W E X Z : C
      c : CategoryTheory.Limits.BinaryCofan W E
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom W X
      h : Quiver.Hom X Z
      i : Quiver.Hom c.pt Z
      H : CategoryTheory.IsPushout f c.inl h i
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY c.inl
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
      ⊢ And (CategoryTheory.IsPullback h' αX αZ h) (CategoryTheory.IsPullback i' αY  …
    -/
  · rintro ⟨hc₃, hc₄⟩
    /-
      case intro.intro.refine_2.mpr.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W E X Z : C
      c : CategoryTheory.Limits.BinaryCofan W E
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      hc : CategoryTheory.Limits.IsColimit c
      f : Quiver.Hom W X
      h : Quiver.Hom X Z
      i : Quiver.Hom c.pt Z
      H : CategoryTheory.IsPushout f c.inl h i
      hc₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
      W' X' Y' Z' : C
      f' : Quiver.Hom W' X'
      g' : Quiver.Hom W' Y'
      h' : Quiver.Hom X' Z'
      i' : Quiver.Hom Y' Z'
      αW : Quiver.Hom W' W
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' (((CategoryTheory.Functor.const (CategoryTheory.Discrete Ca …
      αZ : Quiver.Hom Z' Z
      hf : CategoryTheory.IsPullback f' αW αX f
      hg : CategoryTheory.IsPullback g' αW αY c.inl
      hh : CategoryTheory.CommSq h' αX αZ h
      hi : CategoryTheory.CommSq i' αY αZ i
      w : CategoryTheory.CommSq f' g' h' i'
      hc₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk g' …
      hc₃ : CategoryTheory.IsPullback h' αX αZ h
      hc₄ : CategoryTheory.IsPullback i' αY αZ i
      ⊢ And (CategoryTheory.IsPullback (CategoryTheory.Limits.BinaryCofan.mk (Catego …
    -/
    exact ⟨(IsPullback.of_hasPullback αY c.inr).paste_horiz hc₄, hc₃⟩
    /-
      🎉 no goals
    -/


theorem IsPushout.IsVanKampen.isPullback_of_mono_left [Mono f] {H : IsPushout f g h i}
    (H' : H.IsVanKampen) : IsPullback f g h i :=
  ((H' (𝟙 _) g g (𝟙 Y) (𝟙 _) f (𝟙 _) i (IsKernelPair.id_of_mono f)
                                    /-
                                      C : Type u
                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                      W X Y Z : C
                                      f : Quiver.Hom W X
                                      g : Quiver.Hom W Y
                                      h : Quiver.Hom X Z
                                      i : Quiver.Hom Y Z
                                      inst✝ : CategoryTheory.Mono f
                                      H : CategoryTheory.IsPushout f g h i
                                      H' : H.IsVanKampen
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.id Y …
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
      (IsPullback.of_vert_isIso ⟨by simp⟩) H.1.flip ⟨rfl⟩ ⟨by simp⟩).mp
                                                              /-
                                                                🎉 no goals
                                                              -/
                                  /-
                                    C : Type u
                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                    W X Y Z : C
                                    f : Quiver.Hom W X
                                    g : Quiver.Hom W Y
                                    h : Quiver.Hom X Z
                                    i : Quiver.Hom Y Z
                                    inst✝ : CategoryTheory.Mono f
                                    H : CategoryTheory.IsPushout f g h i
                                    H' : H.IsVanKampen
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id W)  …
                                  -/
    (IsPushout.of_horiz_isIso ⟨by simp⟩)).1.flip
                                  /-
                                    🎉 no goals
                                  -/


theorem IsPushout.IsVanKampen.isPullback_of_mono_right [Mono g] {H : IsPushout f g h i}
    (H' : H.IsVanKampen) : IsPullback f g h i :=
                                                                     /-
                                                                       C : Type u
                                                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                       W X Y Z : C
                                                                       f : Quiver.Hom W X
                                                                       g : Quiver.Hom W Y
                                                                       h : Quiver.Hom X Z
                                                                       i : Quiver.Hom Y Z
                                                                       inst✝ : CategoryTheory.Mono g
                                                                       H : CategoryTheory.IsPushout f g h i
                                                                       H' : H.IsVanKampen
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id X …
                                                                     -/
  ((H' f (𝟙 _) (𝟙 _) f (𝟙 _) (𝟙 _) g h (IsPullback.of_vert_isIso ⟨by simp⟩)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
                                                /-
                                                  C : Type u
                                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                                  W X Y Z : C
                                                  f : Quiver.Hom W X
                                                  g : Quiver.Hom W Y
                                                  h : Quiver.Hom X Z
                                                  i : Quiver.Hom Y Z
                                                  inst✝ : CategoryTheory.Mono g
                                                  H : CategoryTheory.IsPushout f g h i
                                                  H' : H.IsVanKampen
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id X …
                                                -/
      (IsKernelPair.id_of_mono g) ⟨rfl⟩ H.1 ⟨by simp⟩).mp
                                                /-
                                                  🎉 no goals
                                                -/
                                 /-
                                   C : Type u
                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                   W X Y Z : C
                                   f : Quiver.Hom W X
                                   g : Quiver.Hom W Y
                                   h : Quiver.Hom X Z
                                   i : Quiver.Hom Y Z
                                   inst✝ : CategoryTheory.Mono g
                                   H : CategoryTheory.IsPushout f g h i
                                   H' : H.IsVanKampen
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id X …
                                 -/
    (IsPushout.of_vert_isIso ⟨by simp⟩)).2
                                 /-
                                   🎉 no goals
                                 -/


theorem IsPushout.IsVanKampen.mono_of_mono_left [Mono f] {H : IsPushout f g h i}
    (H' : H.IsVanKampen) : Mono i :=
  IsKernelPair.mono_of_isIso_fst
    ((H' (𝟙 _) g g (𝟙 Y) (𝟙 _) f (𝟙 _) i (IsKernelPair.id_of_mono f)
                                      /-
                                        C : Type u
                                        inst✝¹ : CategoryTheory.Category.{v, u} C
                                        W X Y Z : C
                                        f : Quiver.Hom W X
                                        g : Quiver.Hom W Y
                                        h : Quiver.Hom X Z
                                        i : Quiver.Hom Y Z
                                        inst✝ : CategoryTheory.Mono f
                                        H : CategoryTheory.IsPushout f g h i
                                        H' : H.IsVanKampen
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.id Y …
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
        (IsPullback.of_vert_isIso ⟨by simp⟩) H.1.flip ⟨rfl⟩ ⟨by simp⟩).mp
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                    /-
                                      C : Type u
                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                      W X Y Z : C
                                      f : Quiver.Hom W X
                                      g : Quiver.Hom W Y
                                      h : Quiver.Hom X Z
                                      i : Quiver.Hom Y Z
                                      inst✝ : CategoryTheory.Mono f
                                      H : CategoryTheory.IsPushout f g h i
                                      H' : H.IsVanKampen
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id W)  …
                                    -/
      (IsPushout.of_horiz_isIso ⟨by simp⟩)).2
                                    /-
                                      🎉 no goals
                                    -/


theorem IsPushout.IsVanKampen.mono_of_mono_right [Mono g] {H : IsPushout f g h i}
    (H' : H.IsVanKampen) : Mono h :=
  IsKernelPair.mono_of_isIso_fst
                                                                       /-
                                                                         C : Type u
                                                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                         W X Y Z : C
                                                                         f : Quiver.Hom W X
                                                                         g : Quiver.Hom W Y
                                                                         h : Quiver.Hom X Z
                                                                         i : Quiver.Hom Y Z
                                                                         inst✝ : CategoryTheory.Mono g
                                                                         H : CategoryTheory.IsPushout f g h i
                                                                         H' : H.IsVanKampen
                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id X …
                                                                       -/
    ((H' f (𝟙 _) (𝟙 _) f (𝟙 _) (𝟙 _) g h (IsPullback.of_vert_isIso ⟨by simp⟩)
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                  /-
                                                    C : Type u
                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                    W X Y Z : C
                                                    f : Quiver.Hom W X
                                                    g : Quiver.Hom W Y
                                                    h : Quiver.Hom X Z
                                                    i : Quiver.Hom Y Z
                                                    inst✝ : CategoryTheory.Mono g
                                                    H : CategoryTheory.IsPushout f g h i
                                                    H' : H.IsVanKampen
                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id X …
                                                  -/
        (IsKernelPair.id_of_mono g) ⟨rfl⟩ H.1 ⟨by simp⟩).mp
                                                  /-
                                                    🎉 no goals
                                                  -/
                                   /-
                                     C : Type u
                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                     W X Y Z : C
                                     f : Quiver.Hom W X
                                     g : Quiver.Hom W Y
                                     h : Quiver.Hom X Z
                                     i : Quiver.Hom Y Z
                                     inst✝ : CategoryTheory.Mono g
                                     H : CategoryTheory.IsPushout f g h i
                                     H' : H.IsVanKampen
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id X …
                                   -/
      (IsPushout.of_vert_isIso ⟨by simp⟩)).1
                                   /-
                                     🎉 no goals
                                   -/


/-- A category is adhesive if it has pushouts and pullbacks along monomorphisms,
and such pushouts are van Kampen. -/
class Adhesive (C : Type u) [Category.{v} C] : Prop where
  [hasPullback_of_mono_left : ∀ {X Y S : C} (f : X ⟶ S) (g : Y ⟶ S) [Mono f], HasPullback f g]
  [hasPushout_of_mono_left : ∀ {X Y S : C} (f : S ⟶ X) (g : S ⟶ Y) [Mono f], HasPushout f g]
  van_kampen : ∀ {W X Y Z : C} {f : W ⟶ X} {g : W ⟶ Y} {h : X ⟶ Z} {i : Y ⟶ Z} [Mono f]
    (H : IsPushout f g h i), H.IsVanKampen


theorem Adhesive.van_kampen' [Adhesive C] [Mono g] (H : IsPushout f g h i) : H.IsVanKampen :=
  (Adhesive.van_kampen H.flip).flip


theorem Adhesive.isPullback_of_isPushout_of_mono_left [Adhesive C] (H : IsPushout f g h i)
    [Mono f] : IsPullback f g h i :=
  (Adhesive.van_kampen H).isPullback_of_mono_left


theorem Adhesive.isPullback_of_isPushout_of_mono_right [Adhesive C] (H : IsPushout f g h i)
    [Mono g] : IsPullback f g h i :=
  (Adhesive.van_kampen' H).isPullback_of_mono_right


theorem Adhesive.mono_of_isPushout_of_mono_left [Adhesive C] (H : IsPushout f g h i) [Mono f] :
    Mono i :=
  (Adhesive.van_kampen H).mono_of_mono_left


theorem Adhesive.mono_of_isPushout_of_mono_right [Adhesive C] (H : IsPushout f g h i) [Mono g] :
    Mono h :=
  (Adhesive.van_kampen' H).mono_of_mono_right


instance Type.adhesive : Adhesive (Type u) :=
  ⟨fun {_ _ _ _ f _ _ _ _} H =>
    (IsPushout.isVanKampen_inl _ (Types.isCoprodOfMono f) _ _ _ H.flip).flip⟩


noncomputable instance (priority := 100) Adhesive.toRegularMonoCategory [Adhesive C] :
    RegularMonoCategory C :=
  ⟨fun f _ =>
    { Z := pushout f f
      left := pushout.inl _ _
      right := pushout.inr _ _
      w := pushout.condition
      isLimit := (Adhesive.isPullback_of_isPushout_of_mono_left
        (IsPushout.of_hasPushout f f)).isLimitFork }⟩

-- This then implies that adhesive categories are balanced

instance adhesive_functor [Adhesive C] [HasPullbacks C] [HasPushouts C] :
    Adhesive (D ⥤ C) := by
  /-
    J : Type v'
    inst✝⁵ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    inst✝² : CategoryTheory.Adhesive C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    inst✝ : CategoryTheory.Limits.HasPushouts C
    ⊢ CategoryTheory.Adhesive (CategoryTheory.Functor D C)
  -/
  constructor
  /-
    case van_kampen
    J : Type v'
    inst✝⁵ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    inst✝² : CategoryTheory.Adhesive C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    inst✝ : CategoryTheory.Limits.HasPushouts C
    ⊢ ∀ {W X Y Z : CategoryTheory.Functor D C} {f : Quiver.Hom W X} {g : Quiver.Ho …
  -/
  intros W X Y Z f g h i hf H
  /-
    case van_kampen
    J : Type v'
    inst✝⁵ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    W✝ X✝ Y✝ Z✝ : C
    f✝ : Quiver.Hom W✝ X✝
    g✝ : Quiver.Hom W✝ Y✝
    h✝ : Quiver.Hom X✝ Z✝
    i✝ : Quiver.Hom Y✝ Z✝
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    inst✝² : CategoryTheory.Adhesive C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    inst✝ : CategoryTheory.Limits.HasPushouts C
    W X Y Z : CategoryTheory.Functor D C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    ⊢ H.IsVanKampen
  -/
  rw [IsPushout.isVanKampen_iff]
  /-
    case van_kampen
    J : Type v'
    inst✝⁵ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    W✝ X✝ Y✝ Z✝ : C
    f✝ : Quiver.Hom W✝ X✝
    g✝ : Quiver.Hom W✝ Y✝
    h✝ : Quiver.Hom X✝ Z✝
    i✝ : Quiver.Hom Y✝ Z✝
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    inst✝² : CategoryTheory.Adhesive C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    inst✝ : CategoryTheory.Limits.HasPushouts C
    W X Y Z : CategoryTheory.Functor D C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
  -/
  apply isVanKampenColimit_of_evaluation
  /-
    case van_kampen.hc
    J : Type v'
    inst✝⁵ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    W✝ X✝ Y✝ Z✝ : C
    f✝ : Quiver.Hom W✝ X✝
    g✝ : Quiver.Hom W✝ Y✝
    h✝ : Quiver.Hom X✝ Z✝
    i✝ : Quiver.Hom Y✝ Z✝
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    inst✝² : CategoryTheory.Adhesive C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    inst✝ : CategoryTheory.Limits.HasPushouts C
    W X Y Z : CategoryTheory.Functor D C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    ⊢ ∀ (x : D), CategoryTheory.IsVanKampenColimit (((CategoryTheory.evaluation D  …
  -/
  intro x
  /-
    case van_kampen.hc
    J : Type v'
    inst✝⁵ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    W✝ X✝ Y✝ Z✝ : C
    f✝ : Quiver.Hom W✝ X✝
    g✝ : Quiver.Hom W✝ Y✝
    h✝ : Quiver.Hom X✝ Z✝
    i✝ : Quiver.Hom Y✝ Z✝
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    inst✝² : CategoryTheory.Adhesive C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    inst✝ : CategoryTheory.Limits.HasPushouts C
    W X Y Z : CategoryTheory.Functor D C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    x : D
    ⊢ CategoryTheory.IsVanKampenColimit (((CategoryTheory.evaluation D C).obj x).m …
  -/
  refine (IsVanKampenColimit.precompose_isIso_iff (diagramIsoSpan _).inv).mp ?_
  /-
    case van_kampen.hc
    J : Type v'
    inst✝⁵ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    W✝ X✝ Y✝ Z✝ : C
    f✝ : Quiver.Hom W✝ X✝
    g✝ : Quiver.Hom W✝ Y✝
    h✝ : Quiver.Hom X✝ Z✝
    i✝ : Quiver.Hom Y✝ Z✝
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    inst✝² : CategoryTheory.Adhesive C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    inst✝ : CategoryTheory.Limits.HasPushouts C
    W X Y Z : CategoryTheory.Functor D C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    x : D
    ⊢ CategoryTheory.IsVanKampenColimit ((CategoryTheory.Limits.Cocones.precompose …
  -/
  refine IsVanKampenColimit.of_iso ?_ (PushoutCocone.isoMk _).symm
  /-
    case van_kampen.hc
    J : Type v'
    inst✝⁵ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    W✝ X✝ Y✝ Z✝ : C
    f✝ : Quiver.Hom W✝ X✝
    g✝ : Quiver.Hom W✝ Y✝
    h✝ : Quiver.Hom X✝ Z✝
    i✝ : Quiver.Hom Y✝ Z✝
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    inst✝² : CategoryTheory.Adhesive C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    inst✝ : CategoryTheory.Limits.HasPushouts C
    W X Y Z : CategoryTheory.Functor D C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    x : D
    ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk (( …
  -/
  refine (IsPushout.isVanKampen_iff (H.map ((evaluation _ _).obj x))).mp ?_
  /-
    case van_kampen.hc
    J : Type v'
    inst✝⁵ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    W✝ X✝ Y✝ Z✝ : C
    f✝ : Quiver.Hom W✝ X✝
    g✝ : Quiver.Hom W✝ Y✝
    h✝ : Quiver.Hom X✝ Z✝
    i✝ : Quiver.Hom Y✝ Z✝
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    inst✝² : CategoryTheory.Adhesive C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    inst✝ : CategoryTheory.Limits.HasPushouts C
    W X Y Z : CategoryTheory.Functor D C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    x : D
    ⊢ ⋯.IsVanKampen
  -/
  apply Adhesive.van_kampen
  /-
    🎉 no goals
  -/


theorem adhesive_of_preserves_and_reflects (F : C ⥤ D) [Adhesive D]
    [H₁ : ∀ {X Y S : C} (f : X ⟶ S) (g : Y ⟶ S) [Mono f], HasPullback f g]
    [H₂ : ∀ {X Y S : C} (f : S ⟶ X) (g : S ⟶ Y) [Mono f], HasPushout f g]
    [PreservesLimitsOfShape WalkingCospan F]
    [ReflectsLimitsOfShape WalkingCospan F]
    [PreservesColimitsOfShape WalkingSpan F]
    [ReflectsColimitsOfShape WalkingSpan F] :
    Adhesive C := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁵ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁴ : CategoryTheory.Adhesive D
    H₁ : ∀ {X Y S : C} (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : CategoryT …
    H₂ : ∀ {X Y S : C} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    inst✝³ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape CategoryTheory.Limits.Wa …
    ⊢ CategoryTheory.Adhesive C
  -/
  apply Adhesive.mk (hasPullback_of_mono_left := H₁) (hasPushout_of_mono_left := H₂)
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁵ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁴ : CategoryTheory.Adhesive D
    H₁ : ∀ {X Y S : C} (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : CategoryT …
    H₂ : ∀ {X Y S : C} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    inst✝³ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape CategoryTheory.Limits.Wa …
    ⊢ ∀ {W X Y Z : C} {f : Quiver.Hom W X} {g : Quiver.Hom W Y} {h : Quiver.Hom X  …
  -/
  intros W X Y Z f g h i hf H
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁵ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁴ : CategoryTheory.Adhesive D
    H₁ : ∀ {X Y S : C} (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : CategoryT …
    H₂ : ∀ {X Y S : C} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    inst✝³ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape CategoryTheory.Limits.Wa …
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    ⊢ H.IsVanKampen
  -/
  rw [IsPushout.isVanKampen_iff]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁵ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁴ : CategoryTheory.Adhesive D
    H₁ : ∀ {X Y S : C} (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : CategoryT …
    H₂ : ∀ {X Y S : C} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    inst✝³ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape CategoryTheory.Limits.Wa …
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
  -/
  refine IsVanKampenColimit.of_mapCocone F ?_
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁵ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁴ : CategoryTheory.Adhesive D
    H₁ : ∀ {X Y S : C} (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : CategoryT …
    H₂ : ∀ {X Y S : C} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    inst✝³ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape CategoryTheory.Limits.Wa …
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    ⊢ CategoryTheory.IsVanKampenColimit (F.mapCocone (CategoryTheory.Limits.Pushou …
  -/
  refine (IsVanKampenColimit.precompose_isIso_iff (diagramIsoSpan _).inv).mp ?_
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁵ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁴ : CategoryTheory.Adhesive D
    H₁ : ∀ {X Y S : C} (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : CategoryT …
    H₂ : ∀ {X Y S : C} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    inst✝³ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape CategoryTheory.Limits.Wa …
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    ⊢ CategoryTheory.IsVanKampenColimit ((CategoryTheory.Limits.Cocones.precompose …
  -/
  refine IsVanKampenColimit.of_iso ?_ (PushoutCocone.isoMk _).symm
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁵ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁴ : CategoryTheory.Adhesive D
    H₁ : ∀ {X Y S : C} (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : CategoryT …
    H₂ : ∀ {X Y S : C} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    inst✝³ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape CategoryTheory.Limits.Wa …
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone.mk (( …
  -/
  refine (IsPushout.isVanKampen_iff (H.map F)).mp ?_
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁵ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁴ : CategoryTheory.Adhesive D
    H₁ : ∀ {X Y S : C} (f : Quiver.Hom X S) (g : Quiver.Hom Y S) [inst : CategoryT …
    H₂ : ∀ {X Y S : C} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    inst✝³ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape CategoryTheory.Limits.Wa …
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hf : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    ⊢ ⋯.IsVanKampen
  -/
  apply Adhesive.van_kampen
  /-
    🎉 no goals
  -/


theorem adhesive_of_preserves_and_reflects_isomorphism (F : C ⥤ D)
    [Adhesive D] [HasPullbacks C] [HasPushouts C]
    [PreservesLimitsOfShape WalkingCospan F]
    [PreservesColimitsOfShape WalkingSpan F]
    [F.ReflectsIsomorphisms] :
    Adhesive C := by
  haveI : ReflectsLimitsOfShape WalkingCospan F :=
    reflectsLimitsOfShape_of_reflectsIsomorphisms
  haveI : ReflectsColimitsOfShape WalkingSpan F :=
    reflectsColimitsOfShape_of_reflectsIsomorphisms
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁶ : CategoryTheory.Category.{v'', u''} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : CategoryTheory.Adhesive D
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    inst✝³ : CategoryTheory.Limits.HasPushouts C
    inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
    inst✝ : F.ReflectsIsomorphisms
    this✝ : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Walk …
    this : CategoryTheory.Limits.ReflectsColimitsOfShape CategoryTheory.Limits.Wal …
    ⊢ CategoryTheory.Adhesive C
  -/
  exact adhesive_of_preserves_and_reflects F
  /-
    🎉 no goals
  -/


theorem adhesive_of_reflective [HasPullbacks D] [Adhesive C] [HasPullbacks C] [HasPushouts C]
    [H₂ : ∀ {X Y S : D} (f : S ⟶ X) (g : S ⟶ Y) [Mono f], HasPushout f g]
    {Gl : C ⥤ D} {Gr : D ⥤ C} (adj : Gl ⊣ Gr) [Gr.Full] [Gr.Faithful]
    [PreservesLimitsOfShape WalkingCospan Gl] :
    Adhesive D := by
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁷ : CategoryTheory.Category.{v'', u''} D
    inst✝⁶ : CategoryTheory.Limits.HasPullbacks D
    inst✝⁵ : CategoryTheory.Adhesive C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    inst✝³ : CategoryTheory.Limits.HasPushouts C
    H₂ : ∀ {X Y S : D} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝² : Gr.Full
    inst✝¹ : Gr.Faithful
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
    ⊢ CategoryTheory.Adhesive D
  -/
  have := adj.leftAdjoint_preservesColimits
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁷ : CategoryTheory.Category.{v'', u''} D
    inst✝⁶ : CategoryTheory.Limits.HasPullbacks D
    inst✝⁵ : CategoryTheory.Adhesive C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    inst✝³ : CategoryTheory.Limits.HasPushouts C
    H₂ : ∀ {X Y S : D} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝² : Gr.Full
    inst✝¹ : Gr.Faithful
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{?u.117594, ?u.117593, v, …
    ⊢ CategoryTheory.Adhesive D
  -/
  have := adj.rightAdjoint_preservesLimits
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁷ : CategoryTheory.Category.{v'', u''} D
    inst✝⁶ : CategoryTheory.Limits.HasPullbacks D
    inst✝⁵ : CategoryTheory.Adhesive C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    inst✝³ : CategoryTheory.Limits.HasPushouts C
    H₂ : ∀ {X Y S : D} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝² : Gr.Full
    inst✝¹ : Gr.Faithful
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
    this✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{?u.117594, ?u.117593, v …
    this : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.117645, ?u.117644, v'', …
    ⊢ CategoryTheory.Adhesive D
  -/
  apply Adhesive.mk (hasPushout_of_mono_left := H₂)
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁷ : CategoryTheory.Category.{v'', u''} D
    inst✝⁶ : CategoryTheory.Limits.HasPullbacks D
    inst✝⁵ : CategoryTheory.Adhesive C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    inst✝³ : CategoryTheory.Limits.HasPushouts C
    H₂ : ∀ {X Y S : D} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝² : Gr.Full
    inst✝¹ : Gr.Faithful
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
    this✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{?u.117594, ?u.117593, v …
    this : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.117645, ?u.117644, v'', …
    ⊢ ∀ {W X Y Z : D} {f : Quiver.Hom W X} {g : Quiver.Hom W Y} {h : Quiver.Hom X  …
  -/
  intro W X Y Z f g h i _ H
  /-
    C : Type u
    inst✝⁹ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁸ : CategoryTheory.Category.{v'', u''} D
    inst✝⁷ : CategoryTheory.Limits.HasPullbacks D
    inst✝⁶ : CategoryTheory.Adhesive C
    inst✝⁵ : CategoryTheory.Limits.HasPullbacks C
    inst✝⁴ : CategoryTheory.Limits.HasPushouts C
    H₂ : ∀ {X Y S : D} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝³ : Gr.Full
    inst✝² : Gr.Faithful
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    this✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{?u.117594, ?u.117593, v …
    this : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.117645, ?u.117644, v'', …
    W X Y Z : D
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    ⊢ H.IsVanKampen
  -/
  have := Adhesive.van_kampen (IsPushout.of_hasPushout (Gr.map f) (Gr.map g))
  /-
    C : Type u
    inst✝⁹ : CategoryTheory.Category.{v, u} C
    D : Type u''
    inst✝⁸ : CategoryTheory.Category.{v'', u''} D
    inst✝⁷ : CategoryTheory.Limits.HasPullbacks D
    inst✝⁶ : CategoryTheory.Adhesive C
    inst✝⁵ : CategoryTheory.Limits.HasPullbacks C
    inst✝⁴ : CategoryTheory.Limits.HasPushouts C
    H₂ : ∀ {X Y S : D} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝³ : Gr.Full
    inst✝² : Gr.Faithful
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    this✝¹ : CategoryTheory.Limits.PreservesColimitsOfSize.{?u.117594, ?u.117593,  …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, v'', v, u'', u} Gr
    W X Y Z : D
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    h : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Mono f
    H : CategoryTheory.IsPushout f g h i
    this : ⋯.IsVanKampen
    ⊢ H.IsVanKampen
  -/
  rw [IsPushout.isVanKampen_iff] at this ⊢
  refine (IsVanKampenColimit.precompose_isIso_iff
    (isoWhiskerLeft _ (asIso adj.counit) ≪≫ Functor.rightUnitor _).hom).mp ?_
  refine ((this.precompose_isIso (spanCompIso _ _ _).hom).map_reflective adj).of_iso
    (IsColimit.uniqueUpToIso ?_ ?_)
    /-
      case refine_1
      C : Type u
      inst✝⁹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝⁸ : CategoryTheory.Category.{v'', u''} D
      inst✝⁷ : CategoryTheory.Limits.HasPullbacks D
      inst✝⁶ : CategoryTheory.Adhesive C
      inst✝⁵ : CategoryTheory.Limits.HasPullbacks C
      inst✝⁴ : CategoryTheory.Limits.HasPushouts C
      H₂ : ∀ {X Y S : D} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝³ : Gr.Full
      inst✝² : Gr.Faithful
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      this✝¹ : CategoryTheory.Limits.PreservesColimitsOfSize.{?u.117594, ?u.117593,  …
      this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, v'', v, u'', u} Gr
      W X Y Z : D
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Mono f
      H : CategoryTheory.IsPushout f g h i
      this : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone. …
      ⊢ CategoryTheory.Limits.IsColimit (Gl.mapCocone ((CategoryTheory.Limits.Cocone …
    -/
  · exact isColimitOfPreserves Gl ((IsColimit.precomposeHomEquiv _ _).symm <| pushoutIsPushout _ _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝⁹ : CategoryTheory.Category.{v, u} C
      D : Type u''
      inst✝⁸ : CategoryTheory.Category.{v'', u''} D
      inst✝⁷ : CategoryTheory.Limits.HasPullbacks D
      inst✝⁶ : CategoryTheory.Adhesive C
      inst✝⁵ : CategoryTheory.Limits.HasPullbacks C
      inst✝⁴ : CategoryTheory.Limits.HasPushouts C
      H₂ : ∀ {X Y S : D} (f : Quiver.Hom S X) (g : Quiver.Hom S Y) [inst : CategoryT …
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝³ : Gr.Full
      inst✝² : Gr.Faithful
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      this✝¹ : CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v, v'', u, u''} Gl
      this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, v'', v, u'', u} Gr
      W X Y Z : D
      f : Quiver.Hom W X
      g : Quiver.Hom W Y
      h : Quiver.Hom X Z
      i : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Mono f
      H : CategoryTheory.IsPushout f g h i
      this : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.PushoutCocone. …
      ⊢ CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompose ( …
    -/
  · exact (IsColimit.precomposeHomEquiv _ _).symm H.isColimit
    /-
      🎉 no goals
    -/


