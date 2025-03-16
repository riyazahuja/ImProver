/-- The category of partially ordered types. -/
def PartOrd :=
  Bundled PartialOrder


instance : BundledHom.ParentProjection @PartialOrder.toPreorder :=
  ⟨⟩


deriving instance LargeCategory for PartOrd

-- Porting note: probably see https://github.com/leanprover-community/mathlib4/issues/5020

instance : ConcreteCategory PartOrd :=
  BundledHom.concreteCategory _


instance : CoeSort PartOrd Type* :=
  Bundled.coeSort


/-- Construct a bundled PartOrd from the underlying type and typeclass. -/
def of (α : Type*) [PartialOrder α] : PartOrd :=
  Bundled.of α


@[simp]
theorem coe_of (α : Type*) [PartialOrder α] : ↥(of α) = α :=
  rfl


instance : Inhabited PartOrd :=
  ⟨of PUnit⟩


instance (α : PartOrd) : PartialOrder α :=
  α.str


instance hasForgetToPreord : HasForget₂ PartOrd Preord :=
  BundledHom.forget₂ _ _


/-- Constructs an equivalence between partial orders from an order isomorphism between them. -/
@[simps]
def Iso.mk {α β : PartOrd.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : OrderHom α β)
  inv := (e.symm : OrderHom β α)
  hom_inv_id := by
    /-
      α β : PartOrd
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) (CategoryTheory.CategoryS …
    -/
    ext x
    /-
      case w
      α β : PartOrd
      e : OrderIso ↑α ↑β
      x : (CategoryTheory.forget PartOrd).obj α
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) x) ((CategoryTheory.Cate …
    -/
    exact e.symm_apply_apply x
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      α β : PartOrd
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) (CategoryTheory.CategoryS …
    -/
    ext x
    /-
      case w
      α β : PartOrd
      e : OrderIso ↑α ↑β
      x : (CategoryTheory.forget PartOrd).obj β
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) x) ((CategoryTheory.Cate …
    -/
    exact e.apply_symm_apply x
    /-
      🎉 no goals
    -/


/-- `OrderDual` as a functor. -/
@[simps]
def dual : PartOrd ⥤ PartOrd where
  obj X := of Xᵒᵈ
  map := OrderHom.dual


/-- The equivalence between `PartOrd` and itself induced by `OrderDual` both ways. -/
@[simps functor inverse]
def dualEquiv : PartOrd ≌ PartOrd where
  functor := dual
  inverse := dual
             /-
               ⊢ ∀ {X Y : PartOrd} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.co …
             -/
  unitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
             /-
               🎉 no goals
             -/
               /-
                 ⊢ ∀ {X Y : PartOrd} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.co …
               -/
  counitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
               /-
                 🎉 no goals
               -/


theorem partOrd_dual_comp_forget_to_preord :
    PartOrd.dual ⋙ forget₂ PartOrd Preord =
      forget₂ PartOrd Preord ⋙ Preord.dual :=
  rfl


/-- `Antisymmetrization` as a functor. It is the free functor. -/
def preordToPartOrd : Preord.{u} ⥤ PartOrd where
  obj X := PartOrd.of (Antisymmetrization X (· ≤ ·))
  map f := f.antisymmetrization
  map_id X := by
    /-
      X : Preord
      ⊢ Eq ({ obj := fun X => PartOrd.of (Antisymmetrization ↑X fun x1 x2 => LE.le x …
    -/
    ext x
    /-
      case w
      X : Preord
      x : (CategoryTheory.forget PartOrd).obj ({ obj := fun X => PartOrd.of (Antisym …
      ⊢ Eq (({ obj := fun X => PartOrd.of (Antisymmetrization ↑X fun x1 x2 => LE.le  …
    -/
    exact Quotient.inductionOn' x fun x => Quotient.map'_mk'' _ (fun a b => id) _
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      X✝ Y✝ Z✝ : Preord
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => PartOrd.of (Antisymmetrization ↑X fun x1 x2 => LE.le x …
    -/
    ext x
    /-
      case w
      X✝ Y✝ Z✝ : Preord
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      x : (CategoryTheory.forget PartOrd).obj ({ obj := fun X => PartOrd.of (Antisym …
      ⊢ Eq (({ obj := fun X => PartOrd.of (Antisymmetrization ↑X fun x1 x2 => LE.le  …
    -/
    exact Quotient.inductionOn' x fun x => OrderHom.antisymmetrization_apply_mk _ _
    /-
      🎉 no goals
    -/


/-- `preordToPartOrd` is left adjoint to the forgetful functor, meaning it is the free
functor from `Preord` to `PartOrd`. -/
def preordToPartOrdForgetAdjunction :
    preordToPartOrd.{u} ⊣ forget₂ PartOrd Preord :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun _ _ =>
        { toFun := fun f =>
            ⟨f.toFun ∘ toAntisymmetrization (· ≤ ·), f.mono.comp toAntisymmetrization_mono⟩
          invFun := fun f =>
            ⟨fun a => Quotient.liftOn' a f.toFun (fun _ _ h => (AntisymmRel.image h f.mono).eq),
              fun a b => Quotient.inductionOn₂' a b fun _ _ h => f.mono h⟩
          left_inv := fun _ =>
            OrderHom.ext _ _ <| funext fun x => Quotient.inductionOn' x fun _ => rfl
          right_inv := fun _ => OrderHom.ext _ _ <| funext fun _ => rfl }
      homEquiv_naturality_left_symm := fun _ _ =>
        OrderHom.ext _ _ <| funext fun x => Quotient.inductionOn' x fun _ => rfl
      homEquiv_naturality_right := fun _ _ => OrderHom.ext _ _ <| funext fun _ => rfl }

-- The `simpNF` linter would complain as `Functor.comp_obj`, `Preord.dual_obj` both apply to LHS
-- of `preordToPartOrdCompToDualIsoToDualCompPreordToPartOrd_hom_app_coe`

/-- `PreordToPartOrd` and `OrderDual` commute. -/
@[simps! inv_app_coe, simps! (config := .lemmasOnly) hom_app_coe]
def preordToPartOrdCompToDualIsoToDualCompPreordToPartOrd :
    preordToPartOrd.{u} ⋙ PartOrd.dual ≅ Preord.dual ⋙ preordToPartOrd :=
  NatIso.ofComponents (fun _ => PartOrd.Iso.mk <| OrderIso.dualAntisymmetrization _)
    (fun _ => OrderHom.ext _ _ <| funext fun x => Quotient.inductionOn' x fun _ => rfl)

-- This lemma was always bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644

