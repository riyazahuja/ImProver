/-- The category of types equipped with partial functions. -/
def PartialFun : Type _ :=
  Type*


instance : CoeSort PartialFun Type* :=
  ⟨id⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed `@[nolint has_nonempty_instance]`; linter not ported yet

/-- Turns a type into a `PartialFun`. -/
def of : Type* → PartialFun :=
  id


instance : Inhabited PartialFun :=
  ⟨Type*⟩


instance largeCategory : LargeCategory.{u} PartialFun where
  Hom := PFun
  id := PFun.id
  comp f g := g.comp f
  id_comp := @PFun.comp_id
  comp_id := @PFun.id_comp
  assoc _ _ _ := (PFun.comp_assoc _ _ _).symm


/-- Constructs a partial function isomorphism between types from an equivalence between them. -/
@[simps]
def Iso.mk {α β : PartialFun.{u}} (e : α ≃ β) : α ≅ β where
  hom x := e x
  inv x := e.symm x
  hom_inv_id := (PFun.coe_comp _ _).symm.trans (by
    /-
      α β : PartialFun
      e : Equiv α β
      ⊢ Eq (↑(Function.comp ⇑e.symm ⇑e)) (CategoryTheory.CategoryStruct.id α)
    -/
    simp only [Equiv.symm_comp_self, PFun.coe_id]
    /-
      α β : PartialFun
      e : Equiv α β
      ⊢ Eq (PFun.id α) (CategoryTheory.CategoryStruct.id α)
    -/
    rfl)
    /-
      🎉 no goals
    -/
  inv_hom_id := (PFun.coe_comp _ _).symm.trans (by
    /-
      α β : PartialFun
      e : Equiv α β
      ⊢ Eq (↑(Function.comp ⇑e ⇑e.symm)) (CategoryTheory.CategoryStruct.id β)
    -/
    simp only [Equiv.self_comp_symm, PFun.coe_id]
    /-
      α β : PartialFun
      e : Equiv α β
      ⊢ Eq (PFun.id β) (CategoryTheory.CategoryStruct.id β)
    -/
    rfl)
    /-
      🎉 no goals
    -/


/-- The forgetful functor from `Type` to `PartialFun` which forgets that the maps are total. -/
def typeToPartialFun : Type u ⥤ PartialFun where
  obj := id
  map := @PFun.lift
  map_comp _ _ := PFun.coe_comp _ _


instance : typeToPartialFun.Faithful where
  map_injective {_ _} := PFun.lift_injective

-- b ∈ PFun.toSubtype (fun x ↦ x ≠ X.point) Subtype.val a ↔ b ∈ Part.some a

/-- The functor which deletes the point of a pointed type. In return, this makes the maps partial.
This is the computable part of the equivalence `PartialFunEquivPointed`. -/
@[simps obj map]
def pointedToPartialFun : Pointed.{u} ⥤ PartialFun where
  obj X := { x : X // x ≠ X.point }
  map f := PFun.toSubtype _ f.toFun ∘ Subtype.val
  map_id _ :=
    PFun.ext fun _ b =>
      PFun.mem_toSubtype_iff (b := b).trans (Subtype.coe_inj.trans Part.mem_some_iff.symm)
  map_comp f g := by
    -- Porting note: the proof was changed because the original mathlib3 proof no longer works
    /-
      X✝ Y✝ Z✝ : Pointed
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => Subtype fun x => Ne x X.point, map := fun {X Y} f => F …
    -/
    apply PFun.ext _
    /-
      X✝ Y✝ Z✝ : Pointed
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ ∀ (a : { obj := fun X => Subtype fun x => Ne x X.point, map := fun {X Y} f = …
    -/
    rintro ⟨a, ha⟩ ⟨c, hc⟩
    /-
      case mk.mk
      X✝ Y✝ Z✝ : Pointed
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      a : X✝.X
      ha : Ne a X✝.point
      c : Z✝.X
      hc : Ne c Z✝.point
      ⊢ Iff (Membership.mem ({ obj := fun X => Subtype fun x => Ne x X.point, map := …
    -/
    constructor
      /-
        case mk.mk.mp
        X✝ Y✝ Z✝ : Pointed
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        a : X✝.X
        ha : Ne a X✝.point
        c : Z✝.X
        hc : Ne c Z✝.point
        ⊢ Membership.mem ({ obj := fun X => Subtype fun x => Ne x X.point, map := fun  …
      -/
    · rintro ⟨h₁, h₂⟩
      /-
        case mk.mk.mp.intro
        X✝ Y✝ Z✝ : Pointed
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        a : X✝.X
        ha : Ne a X✝.point
        c : Z✝.X
        hc : Ne c Z✝.point
        h₁ : ({ obj := fun X => Subtype fun x => Ne x X.point, map := fun {X Y} f => F …
        h₂ : Eq (({ obj := fun X => Subtype fun x => Ne x X.point, map := fun {X Y} f  …
        ⊢ Membership.mem (CategoryTheory.CategoryStruct.comp ({ obj := fun X => Subtyp …
      -/
      exact ⟨⟨fun h₀ => h₁ ((congr_arg g.toFun h₀).trans g.map_point), h₁⟩, h₂⟩
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.mpr
        X✝ Y✝ Z✝ : Pointed
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        a : X✝.X
        ha : Ne a X✝.point
        c : Z✝.X
        hc : Ne c Z✝.point
        ⊢ Membership.mem (CategoryTheory.CategoryStruct.comp ({ obj := fun X => Subtyp …
      -/
    · rintro ⟨_, _, _⟩
      /-
        case mk.mk.mpr.intro.refl
        X✝ Y✝ Z✝ : Pointed
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        a : X✝.X
        ha : Ne a X✝.point
        w✝ : (CategoryTheory.CategoryStruct.comp ({ obj := fun X => Subtype fun x => N …
        hc : Ne (g.toFun ↑(({ obj := fun X => Subtype fun x => Ne x X.point, map := fu …
        ⊢ Membership.mem ({ obj := fun X => Subtype fun x => Ne x X.point, map := fun  …
      -/
      exact ⟨_, rfl⟩
      /-
        🎉 no goals
      -/


/-- The functor which maps undefined values to a new point. This makes the maps total and creates
pointed types. This is the noncomputable part of the equivalence `PartialFunEquivPointed`. It can't
be computable because `= Option.none` is decidable while the domain of a general `Part` isn't. -/
@[simps obj map]
noncomputable def partialFunToPointed : PartialFun ⥤ Pointed := by
  classical
  exact
    { obj := fun X => ⟨Option X, none⟩
      map := fun f => ⟨Option.elim' none fun a => (f a).toOption, rfl⟩
      map_id := fun X => Pointed.Hom.ext <| funext fun o => Option.recOn o rfl fun a => (by
        dsimp [CategoryStruct.id]
        convert Part.some_toOption a)
      map_comp := fun f g => Pointed.Hom.ext <| funext fun o => Option.recOn o rfl fun a => by
        dsimp [CategoryStruct.comp]
        rw [Part.bind_toOption g (f a), Option.elim'_eq_elim] }


/-- The equivalence induced by `PartialFunToPointed` and `PointedToPartialFun`.
`Part.equivOption` made functorial. -/
@[simps!]
noncomputable def partialFunEquivPointed : PartialFun.{u} ≌ Pointed where
  functor := partialFunToPointed
  inverse := pointedToPartialFun
  unitIso := NatIso.ofComponents (fun X => PartialFun.Iso.mk
      { toFun := fun a => ⟨some a, some_ne_none a⟩
        invFun := fun a => Option.get _ (Option.ne_none_iff_isSome.1 a.2)
        left_inv := fun _ => Option.get_some _ _
                                 /-
                                   X : PartialFun
                                   a : (partialFunToPointed.comp pointedToPartialFun).obj X
                                   ⊢ Eq ((fun a => ⟨Option.some a, ⋯⟩) ((fun a => Option.get ↑a ⋯) a)) a
                                 -/
        right_inv := fun a => by simp only [some_get, Subtype.coe_eta] })
                                 /-
                                   🎉 no goals
                                 -/
      fun f =>
        PFun.ext fun a b => by
          /-
            X✝ Y✝ : PartialFun
            f : Quiver.Hom X✝ Y✝
            a : (CategoryTheory.Functor.id PartialFun).obj X✝
            b : (partialFunToPointed.comp pointedToPartialFun).obj Y✝
            ⊢ Iff (Membership.mem (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Fun …
          -/
          dsimp [PartialFun.Iso.mk, CategoryStruct.comp, pointedToPartialFun]
          /-
            X✝ Y✝ : PartialFun
            f : Quiver.Hom X✝ Y✝
            a : (CategoryTheory.Functor.id PartialFun).obj X✝
            b : (partialFunToPointed.comp pointedToPartialFun).obj Y✝
            ⊢ Iff (Membership.mem ((f a).bind fun x => Part.some ⟨Option.some x, ⋯⟩) b) (M …
          -/
          rw [Part.bind_some]
          -- Porting note: the proof below has changed a lot because
          -- `Part.mem_bind_iff` means that `b ∈ Part.bind f g` is equivalent
          -- to `∃ (a : α), a ∈ f ∧ b ∈ g a`, while in mathlib3 it was equivalent
          -- to `∃ (a : α) (H : a ∈ f), b ∈ g a`
          /-
            X✝ Y✝ : PartialFun
            f : Quiver.Hom X✝ Y✝
            a : (CategoryTheory.Functor.id PartialFun).obj X✝
            b : (partialFunToPointed.comp pointedToPartialFun).obj Y✝
            ⊢ Iff (Membership.mem ((f a).bind fun x => Part.some ⟨Option.some x, ⋯⟩) b) (M …
          -/
          refine (Part.mem_bind_iff.trans ?_).trans PFun.mem_toSubtype_iff.symm
          /-
            X✝ Y✝ : PartialFun
            f : Quiver.Hom X✝ Y✝
            a : (CategoryTheory.Functor.id PartialFun).obj X✝
            b : (partialFunToPointed.comp pointedToPartialFun).obj Y✝
            ⊢ Iff (Exists fun a_1 => And (Membership.mem (f a) a_1) (Membership.mem (Part. …
          -/
          obtain ⟨b | b, hb⟩ := b
            /-
              case mk.none
              X✝ Y✝ : PartialFun
              f : Quiver.Hom X✝ Y✝
              a : (CategoryTheory.Functor.id PartialFun).obj X✝
              hb : Ne Option.none (partialFunToPointed.obj Y✝).point
              ⊢ Iff (Exists fun a_1 => And (Membership.mem (f a) a_1) (Membership.mem (Part. …
            -/
          · exact (hb rfl).elim
            /-
              🎉 no goals
            -/
            /-
              case mk.some
              X✝ Y✝ : PartialFun
              f : Quiver.Hom X✝ Y✝
              a : (CategoryTheory.Functor.id PartialFun).obj X✝
              b : Y✝
              hb : Ne (Option.some b) (partialFunToPointed.obj Y✝).point
              ⊢ Iff (Exists fun a_1 => And (Membership.mem (f a) a_1) (Membership.mem (Part. …
            -/
          · dsimp [Part.toOption]
            /-
              case mk.some
              X✝ Y✝ : PartialFun
              f : Quiver.Hom X✝ Y✝
              a : (CategoryTheory.Functor.id PartialFun).obj X✝
              b : Y✝
              hb : Ne (Option.some b) (partialFunToPointed.obj Y✝).point
              ⊢ Iff (Exists fun a_1 => And (Membership.mem (f a) a_1) (Membership.mem (Part. …
            -/
            simp_rw [Part.mem_some_iff, Subtype.mk_eq_mk]
            /-
              case mk.some
              X✝ Y✝ : PartialFun
              f : Quiver.Hom X✝ Y✝
              a : (CategoryTheory.Functor.id PartialFun).obj X✝
              b : Y✝
              hb : Ne (Option.some b) (partialFunToPointed.obj Y✝).point
              ⊢ Iff (Exists fun a_1 => And (Membership.mem (f a) a_1) (Eq (Option.some b) (O …
            -/
            constructor
              /-
                case mk.some.mp
                X✝ Y✝ : PartialFun
                f : Quiver.Hom X✝ Y✝
                a : (CategoryTheory.Functor.id PartialFun).obj X✝
                b : Y✝
                hb : Ne (Option.some b) (partialFunToPointed.obj Y✝).point
                ⊢ (Exists fun a_1 => And (Membership.mem (f a) a_1) (Eq (Option.some b) (Optio …
              -/
            · rintro ⟨_, ⟨h₁, h₂⟩, h₃⟩
              /-
                case mk.some.mp.intro.intro.intro
                X✝ Y✝ : PartialFun
                f : Quiver.Hom X✝ Y✝
                a : (CategoryTheory.Functor.id PartialFun).obj X✝
                b : Y✝
                hb : Ne (Option.some b) (partialFunToPointed.obj Y✝).point
                w✝ : Y✝
                h₃ : Eq (Option.some b) (Option.some w✝)
                h₁ : (f a).Dom
                h₂ : Eq ((f a).get h₁) w✝
                ⊢ Eq (Option.some b) (dite (f a).Dom (fun h => Option.some ((f a).get h)) fun  …
              -/
              rw [h₃, ← h₂, dif_pos h₁]
              /-
                🎉 no goals
              -/
              /-
                case mk.some.mpr
                X✝ Y✝ : PartialFun
                f : Quiver.Hom X✝ Y✝
                a : (CategoryTheory.Functor.id PartialFun).obj X✝
                b : Y✝
                hb : Ne (Option.some b) (partialFunToPointed.obj Y✝).point
                ⊢ Eq (Option.some b) (dite (f a).Dom (fun h => Option.some ((f a).get h)) fun  …
              -/
            · intro h
              /-
                case mk.some.mpr
                X✝ Y✝ : PartialFun
                f : Quiver.Hom X✝ Y✝
                a : (CategoryTheory.Functor.id PartialFun).obj X✝
                b : Y✝
                hb : Ne (Option.some b) (partialFunToPointed.obj Y✝).point
                h : Eq (Option.some b) (dite (f a).Dom (fun h => Option.some ((f a).get h)) fu …
                ⊢ Exists fun a_1 => And (Membership.mem (f a) a_1) (Eq (Option.some b) (Option …
              -/
              split_ifs at h with ha
              /-
                case pos
                X✝ Y✝ : PartialFun
                f : Quiver.Hom X✝ Y✝
                a : (CategoryTheory.Functor.id PartialFun).obj X✝
                b : Y✝
                hb : Ne (Option.some b) (partialFunToPointed.obj Y✝).point
                ha : (f a).Dom
                h : Eq (Option.some b) (Option.some ((f a).get ha))
                ⊢ Exists fun a_1 => And (Membership.mem (f a) a_1) (Eq (Option.some b) (Option …
              -/
              rw [some_inj] at h
              /-
                case pos
                X✝ Y✝ : PartialFun
                f : Quiver.Hom X✝ Y✝
                a : (CategoryTheory.Functor.id PartialFun).obj X✝
                b : Y✝
                hb : Ne (Option.some b) (partialFunToPointed.obj Y✝).point
                ha : (f a).Dom
                h : Eq b ((f a).get ha)
                ⊢ Exists fun a_1 => And (Membership.mem (f a) a_1) (Eq (Option.some b) (Option …
              -/
              exact ⟨b, ⟨ha, h.symm⟩, rfl⟩
              /-
                🎉 no goals
              -/
  counitIso :=
    NatIso.ofComponents
                                  /-
                                    X : Pointed
                                    ⊢ Equiv ((pointedToPartialFun.comp partialFunToPointed).obj X).X ((CategoryThe …
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
      (fun X ↦ Pointed.Iso.mk (by classical exact Equiv.optionSubtypeNe X.point) (by rfl))
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
      fun {X Y} f ↦ Pointed.Hom.ext <| funext fun a ↦ by
        /-
          X Y : Pointed
          f : Quiver.Hom X Y
          a : ((pointedToPartialFun.comp partialFunToPointed).obj X).X
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((pointedToPartialFun.comp partialFu …
        -/
        obtain _ | ⟨a, ha⟩ := a
          /-
            case none
            X Y : Pointed
            f : Quiver.Hom X Y
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((pointedToPartialFun.comp partialFu …
          -/
        · exact f.map_point.symm
          /-
            🎉 no goals
          -/
        /-
          case some.mk
          X Y : Pointed
          f : Quiver.Hom X Y
          a : X.X
          ha : Ne a X.point
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((pointedToPartialFun.comp partialFu …
        -/
        simp_all [Option.casesOn'_eq_elim, Part.elim_toOption]
        /-
          🎉 no goals
        -/
  functor_unitIso_comp X := by
    /-
      X : PartialFun
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (partialFunToPointed.map ((CategoryTh …
    -/
    ext (_ | x)
      /-
        case w.none
        X : PartialFun
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (partialFunToPointed.map ((CategoryT …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case w.some
        X : PartialFun
        x : (CategoryTheory.Functor.id PartialFun).obj X
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (partialFunToPointed.map ((CategoryT …
      -/
    · simp
      /-
        case w.some
        X : PartialFun
        x : (CategoryTheory.Functor.id PartialFun).obj X
        ⊢ Eq ((Pointed.Iso.mk (Equiv.optionSubtypeNe Option.none) ⋯).hom ({ toFun := O …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- Forgetting that maps are total and making them total again by adding a point is the same as just
adding a point. -/
@[simps!]
noncomputable def typeToPartialFunIsoPartialFunToPointed :
    typeToPartialFun ⋙ partialFunToPointed ≅ typeToPointed :=
  NatIso.ofComponents
    (fun _ =>
      { hom := ⟨id, rfl⟩
        inv := ⟨id, rfl⟩
        hom_inv_id := rfl
        inv_hom_id := rfl })
    fun f =>
    Pointed.Hom.ext <|
      funext fun a => Option.recOn a rfl fun a => by
        /-
          X✝ Y✝ : Type ?u.14763
          f : Quiver.Hom X✝ Y✝
          a✝ : ((typeToPartialFun.comp partialFunToPointed).obj X✝).X
          a : typeToPartialFun.obj X✝
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((typeToPartialFun.comp partialFunTo …
        -/
        convert Part.some_toOption _
        /-
          case h.e'_2
          X✝ Y✝ : Type ?u.14763
          f : Quiver.Hom X✝ Y✝
          a✝ : ((typeToPartialFun.comp partialFunToPointed).obj X✝).X
          a : typeToPartialFun.obj X✝
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((typeToPartialFun.comp partialFunTo …
        -/
        simpa using (Part.get_eq_iff_mem (by trivial)).mp rfl
        /-
          🎉 no goals
        -/

