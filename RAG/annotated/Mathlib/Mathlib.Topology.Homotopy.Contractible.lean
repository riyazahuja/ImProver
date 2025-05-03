/-- A map is nullhomotopic if it is homotopic to a constant map. -/
def Nullhomotopic (f : C(X, Y)) : Prop :=
  ∃ y : Y, Homotopic f (ContinuousMap.const _ y)


theorem nullhomotopic_of_constant (y : Y) : Nullhomotopic (ContinuousMap.const X y) :=
         /-
           X : Type u_1
           Y : Type u_2
           inst✝¹ : TopologicalSpace X
           inst✝ : TopologicalSpace Y
           y : Y
           ⊢ (ContinuousMap.const X y).Homotopic (ContinuousMap.const X y)
         -/
  ⟨y, by rfl⟩
         /-
           🎉 no goals
         -/


theorem Nullhomotopic.comp_right {f : C(X, Y)} (hf : f.Nullhomotopic) (g : C(Y, Z)) :
    (g.comp f).Nullhomotopic := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : ContinuousMap X Y
    hf : f.Nullhomotopic
    g : ContinuousMap Y Z
    ⊢ (g.comp f).Nullhomotopic
  -/
  cases' hf with y hy
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : ContinuousMap X Y
    g : ContinuousMap Y Z
    y : Y
    hy : f.Homotopic (ContinuousMap.const X y)
    ⊢ (g.comp f).Nullhomotopic
  -/
  use g y
  /-
    case h
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : ContinuousMap X Y
    g : ContinuousMap Y Z
    y : Y
    hy : f.Homotopic (ContinuousMap.const X y)
    ⊢ (g.comp f).Homotopic (ContinuousMap.const X (g y))
  -/
  exact Homotopic.hcomp hy (Homotopic.refl g)
  /-
    🎉 no goals
  -/


theorem Nullhomotopic.comp_left {f : C(Y, Z)} (hf : f.Nullhomotopic) (g : C(X, Y)) :
    (f.comp g).Nullhomotopic := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : ContinuousMap Y Z
    hf : f.Nullhomotopic
    g : ContinuousMap X Y
    ⊢ (f.comp g).Nullhomotopic
  -/
  cases' hf with y hy
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : ContinuousMap Y Z
    g : ContinuousMap X Y
    y : Z
    hy : f.Homotopic (ContinuousMap.const Y y)
    ⊢ (f.comp g).Nullhomotopic
  -/
  use y
  /-
    case h
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : ContinuousMap Y Z
    g : ContinuousMap X Y
    y : Z
    hy : f.Homotopic (ContinuousMap.const Y y)
    ⊢ (f.comp g).Homotopic (ContinuousMap.const X y)
  -/
  exact Homotopic.hcomp (Homotopic.refl g) hy
  /-
    🎉 no goals
  -/


/-- A contractible space is one that is homotopy equivalent to `Unit`. -/
class ContractibleSpace (X : Type*) [TopologicalSpace X] : Prop where
  hequiv_unit' : Nonempty (X ≃ₕ Unit)

-- Porting note: added to work around lack of infer kinds

theorem ContractibleSpace.hequiv_unit (X : Type*) [TopologicalSpace X] [ContractibleSpace X] :
    Nonempty (X ≃ₕ Unit) :=
  ContractibleSpace.hequiv_unit'


theorem id_nullhomotopic (X : Type*) [TopologicalSpace X] [ContractibleSpace X] :
    (ContinuousMap.id X).Nullhomotopic := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : ContractibleSpace X
    ⊢ (ContinuousMap.id X).Nullhomotopic
  -/
  obtain ⟨hv⟩ := ContractibleSpace.hequiv_unit X
  /-
    case intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : ContractibleSpace X
    hv : ContinuousMap.HomotopyEquiv X Unit
    ⊢ (ContinuousMap.id X).Nullhomotopic
  -/
  use hv.invFun ()
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : ContractibleSpace X
    hv : ContinuousMap.HomotopyEquiv X Unit
    ⊢ (ContinuousMap.id X).Homotopic (ContinuousMap.const X (hv.invFun Unit.unit))
  -/
  convert hv.left_inv.symm
  /-
    🎉 no goals
  -/


theorem contractible_iff_id_nullhomotopic (Y : Type*) [TopologicalSpace Y] :
    ContractibleSpace Y ↔ (ContinuousMap.id Y).Nullhomotopic := by
  /-
    Y : Type u_1
    inst✝ : TopologicalSpace Y
    ⊢ Iff (ContractibleSpace Y) (ContinuousMap.id Y).Nullhomotopic
  -/
  constructor
    /-
      case mp
      Y : Type u_1
      inst✝ : TopologicalSpace Y
      ⊢ ContractibleSpace Y → (ContinuousMap.id Y).Nullhomotopic
    -/
  · intro
    /-
      case mp
      Y : Type u_1
      inst✝ : TopologicalSpace Y
      a✝ : ContractibleSpace Y
      ⊢ (ContinuousMap.id Y).Nullhomotopic
    -/
    apply id_nullhomotopic
    /-
      🎉 no goals
    -/
  /-
    case mpr
    Y : Type u_1
    inst✝ : TopologicalSpace Y
    ⊢ (ContinuousMap.id Y).Nullhomotopic → ContractibleSpace Y
  -/
  rintro ⟨p, h⟩
  refine
    { hequiv_unit' :=
        ⟨{  toFun := ContinuousMap.const _ ()
            invFun := ContinuousMap.const _ p
            left_inv := ?_
            right_inv := ?_ }⟩ }
    /-
      case mpr.intro.refine_1
      Y : Type u_1
      inst✝ : TopologicalSpace Y
      p : Y
      h : (ContinuousMap.id Y).Homotopic (ContinuousMap.const Y p)
      ⊢ ((ContinuousMap.const Unit p).comp (ContinuousMap.const Y Unit.unit)).Homoto …
    -/
  · exact h.symm
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro.refine_2
      Y : Type u_1
      inst✝ : TopologicalSpace Y
      p : Y
      h : (ContinuousMap.id Y).Homotopic (ContinuousMap.const Y p)
      ⊢ ((ContinuousMap.const Y Unit.unit).comp (ContinuousMap.const Unit p)).Homoto …
    -/
  · convert Homotopic.refl (ContinuousMap.id Unit)
    /-
      🎉 no goals
    -/


protected theorem ContinuousMap.HomotopyEquiv.contractibleSpace [ContractibleSpace Y] (e : X ≃ₕ Y) :
    ContractibleSpace X :=
  ⟨(ContractibleSpace.hequiv_unit Y).map e.trans⟩


protected theorem ContinuousMap.HomotopyEquiv.contractibleSpace_iff (e : X ≃ₕ Y) :
    ContractibleSpace X ↔ ContractibleSpace Y :=
  ⟨fun _ => e.symm.contractibleSpace, fun _ => e.contractibleSpace⟩


protected theorem Homeomorph.contractibleSpace [ContractibleSpace Y] (e : X ≃ₜ Y) :
    ContractibleSpace X :=
  e.toHomotopyEquiv.contractibleSpace


protected theorem Homeomorph.contractibleSpace_iff (e : X ≃ₜ Y) :
    ContractibleSpace X ↔ ContractibleSpace Y :=
  e.toHomotopyEquiv.contractibleSpace_iff


instance [Nonempty Y] [Subsingleton Y] : ContractibleSpace Y :=
  let ⟨_⟩ := nonempty_unique Y
  ⟨⟨(Homeomorph.homeomorphOfUnique Y Unit).toHomotopyEquiv⟩⟩


variable (X Y) in
theorem hequiv [ContractibleSpace X] [ContractibleSpace Y] :
    Nonempty (X ≃ₕ Y) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : ContractibleSpace X
    inst✝ : ContractibleSpace Y
    ⊢ Nonempty (ContinuousMap.HomotopyEquiv X Y)
  -/
  rcases ContractibleSpace.hequiv_unit' (X := X) with ⟨h⟩
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : ContractibleSpace X
    inst✝ : ContractibleSpace Y
    h : ContinuousMap.HomotopyEquiv X Unit
    ⊢ Nonempty (ContinuousMap.HomotopyEquiv X Y)
  -/
  rcases ContractibleSpace.hequiv_unit' (X := Y) with ⟨h'⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : ContractibleSpace X
    inst✝ : ContractibleSpace Y
    h : ContinuousMap.HomotopyEquiv X Unit
    h' : ContinuousMap.HomotopyEquiv Y Unit
    ⊢ Nonempty (ContinuousMap.HomotopyEquiv X Y)
  -/
  exact ⟨h.trans h'.symm⟩
  /-
    🎉 no goals
  -/


instance (priority := 100) [ContractibleSpace X] : PathConnectedSpace X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : ContractibleSpace X
    ⊢ PathConnectedSpace X
  -/
  obtain ⟨p, ⟨h⟩⟩ := id_nullhomotopic X
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : ContractibleSpace X
    p : X
    h : (ContinuousMap.id X).Homotopy (ContinuousMap.const X p)
    ⊢ PathConnectedSpace X
  -/
  have : ∀ x, Joined p x := fun x => ⟨(h.evalAt x).symm⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : ContractibleSpace X
    p : X
    h : (ContinuousMap.id X).Homotopy (ContinuousMap.const X p)
    this : ∀ (x : X), Joined p x
    ⊢ PathConnectedSpace X
  -/
  rw [pathConnectedSpace_iff_eq]; use p; ext; tauto
                                              /-
                                                🎉 no goals
                                              -/


