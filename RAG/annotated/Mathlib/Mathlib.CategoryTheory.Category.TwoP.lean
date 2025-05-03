/-- The category of two-pointed types. -/
structure TwoP : Type (u + 1) where
  /-- The underlying type of a two-pointed type. -/
  protected X : Type u
  /-- The two points of a bipointed type, bundled together as a pair of distinct elements. -/
  toTwoPointing : TwoPointing X


instance : CoeSort TwoP Type* :=
  ⟨TwoP.X⟩


/-- Turns a two-pointing into a two-pointed type. -/
def of {X : Type*} (toTwoPointing : TwoPointing X) : TwoP :=
  ⟨X, toTwoPointing⟩


@[simp]
theorem coe_of {X : Type*} (toTwoPointing : TwoPointing X) : ↥(of toTwoPointing) = X :=
  rfl


alias _root_.TwoPointing.TwoP := of


instance : Inhabited TwoP :=
  ⟨of TwoPointing.bool⟩


/-- Turns a two-pointed type into a bipointed type, by forgetting that the pointed elements are
distinct. -/
noncomputable def toBipointed (X : TwoP) : Bipointed :=
  X.toTwoPointing.toProd.Bipointed


@[simp]
theorem coe_toBipointed (X : TwoP) : ↥X.toBipointed = ↥X :=
  rfl


noncomputable instance largeCategory : LargeCategory TwoP :=
  InducedCategory.category toBipointed


noncomputable instance concreteCategory : ConcreteCategory TwoP :=
  InducedCategory.concreteCategory toBipointed


noncomputable instance hasForgetToBipointed : HasForget₂ TwoP Bipointed :=
  InducedCategory.hasForget₂ toBipointed



/-- Swaps the pointed elements of a two-pointed type. `TwoPointing.swap` as a functor. -/
@[simps]
noncomputable def swap : TwoP ⥤ TwoP where
  obj X := ⟨X, X.toTwoPointing.swap⟩
  map f := ⟨f.toFun, f.map_snd, f.map_fst⟩


/-- The equivalence between `TwoP` and itself induced by `Prod.swap` both ways. -/
@[simps!]
noncomputable def swapEquiv : TwoP ≌ TwoP where
  functor := swap
  inverse := swap
  unitIso := Iso.refl _
  counitIso := Iso.refl _


@[simp]
theorem swapEquiv_symm : swapEquiv.symm = swapEquiv :=
  rfl


@[simp]
theorem TwoP_swap_comp_forget_to_Bipointed :
    TwoP.swap ⋙ forget₂ TwoP Bipointed = forget₂ TwoP Bipointed ⋙ Bipointed.swap :=
  rfl


/-- The functor from `Pointed` to `TwoP` which adds a second point. -/
@[simps]
noncomputable def pointedToTwoPFst : Pointed.{u} ⥤ TwoP where
  obj X := ⟨Option X, ⟨X.point, none⟩, some_ne_none _⟩
  map f := ⟨Option.map f.toFun, congr_arg _ f.map_point, rfl⟩
  map_id _ := Bipointed.Hom.ext Option.map_id
  map_comp f g := Bipointed.Hom.ext (Option.map_comp_map f.1 g.1).symm


/-- The functor from `Pointed` to `TwoP` which adds a first point. -/
@[simps]
noncomputable def pointedToTwoPSnd : Pointed.{u} ⥤ TwoP where
  obj X := ⟨Option X, ⟨none, X.point⟩, (some_ne_none _).symm⟩
  map f := ⟨Option.map f.toFun, rfl, congr_arg _ f.map_point⟩
  map_id _ := Bipointed.Hom.ext Option.map_id
  map_comp f g := Bipointed.Hom.ext (Option.map_comp_map f.1 g.1).symm


@[simp]
theorem pointedToTwoPFst_comp_swap : pointedToTwoPFst ⋙ TwoP.swap = pointedToTwoPSnd :=
  rfl


@[simp]
theorem pointedToTwoPSnd_comp_swap : pointedToTwoPSnd ⋙ TwoP.swap = pointedToTwoPFst :=
  rfl


@[simp]
theorem pointedToTwoPFst_comp_forget_to_bipointed :
    pointedToTwoPFst ⋙ forget₂ TwoP Bipointed = pointedToBipointedFst :=
  rfl


@[simp]
theorem pointedToTwoPSnd_comp_forget_to_bipointed :
    pointedToTwoPSnd ⋙ forget₂ TwoP Bipointed = pointedToBipointedSnd :=
  rfl


/-- Adding a second point is left adjoint to forgetting the second point. -/
noncomputable def pointedToTwoPFstForgetCompBipointedToPointedFstAdjunction :
    pointedToTwoPFst ⊣ forget₂ TwoP Bipointed ⋙ bipointedToPointedFst :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun X Y =>
        { toFun := fun f => ⟨f.toFun ∘ Option.some, f.map_fst⟩
          invFun := fun f => ⟨fun o => o.elim Y.toTwoPointing.toProd.2 f.toFun, f.map_point, rfl⟩
          left_inv := fun f => by
            /-
              α : Type u_1
              β : Type u_2
              X : Pointed
              Y : TwoP
              f : Quiver.Hom (pointedToTwoPFst.obj X) Y
              ⊢ Eq ((fun f => { toFun := fun o => Option.elim o Y.toTwoPointing.toProd.2 f.t …
            -/
            apply Bipointed.Hom.ext
            /-
              case toFun
              α : Type u_1
              β : Type u_2
              X : Pointed
              Y : TwoP
              f : Quiver.Hom (pointedToTwoPFst.obj X) Y
              ⊢ Eq ((fun f => { toFun := fun o => Option.elim o Y.toTwoPointing.toProd.2 f.t …
            -/
            funext x
            /-
              case toFun.h
              α : Type u_1
              β : Type u_2
              X : Pointed
              Y : TwoP
              f : Quiver.Hom (pointedToTwoPFst.obj X) Y
              x : (pointedToTwoPFst.obj X).toBipointed.X
              ⊢ Eq (((fun f => { toFun := fun o => Option.elim o Y.toTwoPointing.toProd.2 f. …
            -/
            cases x
              /-
                case toFun.h.none
                α : Type u_1
                β : Type u_2
                X : Pointed
                Y : TwoP
                f : Quiver.Hom (pointedToTwoPFst.obj X) Y
                ⊢ Eq (((fun f => { toFun := fun o => Option.elim o Y.toTwoPointing.toProd.2 f. …
              -/
            · exact f.map_snd.symm
              /-
                🎉 no goals
              -/
              /-
                case toFun.h.some
                α : Type u_1
                β : Type u_2
                X : Pointed
                Y : TwoP
                f : Quiver.Hom (pointedToTwoPFst.obj X) Y
                val✝ : X.X
                ⊢ Eq (((fun f => { toFun := fun o => Option.elim o Y.toTwoPointing.toProd.2 f. …
              -/
            · rfl
              /-
                🎉 no goals
              -/
          right_inv := fun _ => Pointed.Hom.ext rfl }
      homEquiv_naturality_left_symm := fun f g => by
        /-
          α : Type u_1
          β : Type u_2
          X'✝ X✝ : Pointed
          Y✝ : TwoP
          f : Quiver.Hom X'✝ X✝
          g : Quiver.Hom X✝ (((CategoryTheory.forget₂ TwoP Bipointed).comp bipointedToPo …
          ⊢ Eq (((fun X Y => { toFun := fun f => { toFun := Function.comp f.toFun Option …
        -/
        apply Bipointed.Hom.ext
        /-
          case toFun
          α : Type u_1
          β : Type u_2
          X'✝ X✝ : Pointed
          Y✝ : TwoP
          f : Quiver.Hom X'✝ X✝
          g : Quiver.Hom X✝ (((CategoryTheory.forget₂ TwoP Bipointed).comp bipointedToPo …
          ⊢ Eq (((fun X Y => { toFun := fun f => { toFun := Function.comp f.toFun Option …
        -/
        funext x
        /-
          case toFun.h
          α : Type u_1
          β : Type u_2
          X'✝ X✝ : Pointed
          Y✝ : TwoP
          f : Quiver.Hom X'✝ X✝
          g : Quiver.Hom X✝ (((CategoryTheory.forget₂ TwoP Bipointed).comp bipointedToPo …
          x : (pointedToTwoPFst.obj X'✝).toBipointed.X
          ⊢ Eq ((((fun X Y => { toFun := fun f => { toFun := Function.comp f.toFun Optio …
        -/
                    /-
                      🎉 no goals
                    -/
        cases x <;> rfl }
                    /-
                      🎉 no goals
                    -/


/-- Adding a first point is left adjoint to forgetting the first point. -/
noncomputable def pointedToTwoPSndForgetCompBipointedToPointedSndAdjunction :
    pointedToTwoPSnd ⊣ forget₂ TwoP Bipointed ⋙ bipointedToPointedSnd :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun X Y =>
        { toFun := fun f => ⟨f.toFun ∘ Option.some, f.map_snd⟩
          invFun := fun f => ⟨fun o => o.elim Y.toTwoPointing.toProd.1 f.toFun, rfl, f.map_point⟩
          left_inv := fun f => by
            /-
              α : Type u_1
              β : Type u_2
              X : Pointed
              Y : TwoP
              f : Quiver.Hom (pointedToTwoPSnd.obj X) Y
              ⊢ Eq ((fun f => { toFun := fun o => Option.elim o Y.toTwoPointing.toProd.1 f.t …
            -/
            apply Bipointed.Hom.ext
            /-
              case toFun
              α : Type u_1
              β : Type u_2
              X : Pointed
              Y : TwoP
              f : Quiver.Hom (pointedToTwoPSnd.obj X) Y
              ⊢ Eq ((fun f => { toFun := fun o => Option.elim o Y.toTwoPointing.toProd.1 f.t …
            -/
            funext x
            /-
              case toFun.h
              α : Type u_1
              β : Type u_2
              X : Pointed
              Y : TwoP
              f : Quiver.Hom (pointedToTwoPSnd.obj X) Y
              x : (pointedToTwoPSnd.obj X).toBipointed.X
              ⊢ Eq (((fun f => { toFun := fun o => Option.elim o Y.toTwoPointing.toProd.1 f. …
            -/
            cases x
              /-
                case toFun.h.none
                α : Type u_1
                β : Type u_2
                X : Pointed
                Y : TwoP
                f : Quiver.Hom (pointedToTwoPSnd.obj X) Y
                ⊢ Eq (((fun f => { toFun := fun o => Option.elim o Y.toTwoPointing.toProd.1 f. …
              -/
            · exact f.map_fst.symm
              /-
                🎉 no goals
              -/
              /-
                case toFun.h.some
                α : Type u_1
                β : Type u_2
                X : Pointed
                Y : TwoP
                f : Quiver.Hom (pointedToTwoPSnd.obj X) Y
                val✝ : X.X
                ⊢ Eq (((fun f => { toFun := fun o => Option.elim o Y.toTwoPointing.toProd.1 f. …
              -/
            · rfl
              /-
                🎉 no goals
              -/
          right_inv := fun _ => Pointed.Hom.ext rfl }
      homEquiv_naturality_left_symm := fun f g => by
        /-
          α : Type u_1
          β : Type u_2
          X'✝ X✝ : Pointed
          Y✝ : TwoP
          f : Quiver.Hom X'✝ X✝
          g : Quiver.Hom X✝ (((CategoryTheory.forget₂ TwoP Bipointed).comp bipointedToPo …
          ⊢ Eq (((fun X Y => { toFun := fun f => { toFun := Function.comp f.toFun Option …
        -/
        apply Bipointed.Hom.ext
        /-
          case toFun
          α : Type u_1
          β : Type u_2
          X'✝ X✝ : Pointed
          Y✝ : TwoP
          f : Quiver.Hom X'✝ X✝
          g : Quiver.Hom X✝ (((CategoryTheory.forget₂ TwoP Bipointed).comp bipointedToPo …
          ⊢ Eq (((fun X Y => { toFun := fun f => { toFun := Function.comp f.toFun Option …
        -/
        funext x
        /-
          case toFun.h
          α : Type u_1
          β : Type u_2
          X'✝ X✝ : Pointed
          Y✝ : TwoP
          f : Quiver.Hom X'✝ X✝
          g : Quiver.Hom X✝ (((CategoryTheory.forget₂ TwoP Bipointed).comp bipointedToPo …
          x : (pointedToTwoPSnd.obj X'✝).toBipointed.X
          ⊢ Eq ((((fun X Y => { toFun := fun f => { toFun := Function.comp f.toFun Optio …
        -/
                    /-
                      🎉 no goals
                    -/
        cases x <;> rfl }
                    /-
                      🎉 no goals
                    -/

