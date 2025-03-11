/-- A simply connected space is one whose fundamental groupoid is equivalent to `Discrete Unit` -/
@[mk_iff simply_connected_def]
class SimplyConnectedSpace (X : Type*) [TopologicalSpace X] : Prop where
  equiv_unit : Nonempty (FundamentalGroupoid X ≌ Discrete Unit)


theorem simply_connected_iff_unique_homotopic (X : Type*) [TopologicalSpace X] :
    SimplyConnectedSpace X ↔
      Nonempty X ∧ ∀ x y : X, Nonempty (Unique (Path.Homotopic.Quotient x y)) := by
  simp only [simply_connected_def, equiv_punit_iff_unique,
    FundamentalGroupoid.nonempty_iff X, and_congr_right_iff, Nonempty.forall]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ X → Iff (∀ (x y : FundamentalGroupoid X), Nonempty (Unique (Quiver.Hom x y)) …
  -/
  intros
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a✝ : X
    ⊢ Iff (∀ (x y : FundamentalGroupoid X), Nonempty (Unique (Quiver.Hom x y))) (∀ …
  -/
  exact ⟨fun h _ _ => h _ _, fun h _ _ => h _ _⟩
  /-
    🎉 no goals
  -/


instance (x y : X) : Subsingleton (Path.Homotopic.Quotient x y) :=
  @Unique.instSubsingleton _ (Nonempty.some (by
    /-
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : SimplyConnectedSpace X
      x y : X
      ⊢ Nonempty (Unique (Path.Homotopic.Quotient x y))
    -/
    rw [simply_connected_iff_unique_homotopic] at *; tauto))
                                                     /-
                                                       🎉 no goals
                                                     -/


instance (priority := 100) : PathConnectedSpace X :=
  let unique_homotopic := (simply_connected_iff_unique_homotopic X).mp inferInstance
  { nonempty := unique_homotopic.1
    joined := fun x y => ⟨(unique_homotopic.2 x y).some.default.out⟩ }


/-- In a simply connected space, any two paths are homotopic -/
theorem paths_homotopic {x y : X} (p₁ p₂ : Path x y) : Path.Homotopic p₁ p₂ :=
  Quotient.eq.mp (@Subsingleton.elim (Path.Homotopic.Quotient x y) _ _ _)


instance (priority := 100) ofContractible (Y : Type u) [TopologicalSpace Y] [ContractibleSpace Y] :
    SimplyConnectedSpace Y where
  equiv_unit :=
    let H : TopCat.of Y ≃ₕ TopCat.of PUnit.{u+1} := (ContractibleSpace.hequiv Y PUnit.{u+1}).some
    ⟨(FundamentalGroupoidFunctor.equivOfHomotopyEquiv H).trans
      FundamentalGroupoid.punitEquivDiscretePUnit⟩


/-- A space is simply connected iff it is path connected, and there is at most one path
  up to homotopy between any two points. -/
theorem simply_connected_iff_paths_homotopic {Y : Type*} [TopologicalSpace Y] :
    SimplyConnectedSpace Y ↔
      PathConnectedSpace Y ∧ ∀ x y : Y, Subsingleton (Path.Homotopic.Quotient x y) :=
      /-
        Y : Type u_1
        inst✝ : TopologicalSpace Y
        ⊢ SimplyConnectedSpace Y → And (PathConnectedSpace Y) (∀ (x y : Y), Subsinglet …
      -/
                             /-
                               🎉 no goals
                             -/
  ⟨by intro; constructor <;> infer_instance, fun h => by
                             /-
                               🎉 no goals
                             -/
    /-
      Y : Type u_1
      inst✝ : TopologicalSpace Y
      h : And (PathConnectedSpace Y) (∀ (x y : Y), Subsingleton (Path.Homotopic.Quot …
      ⊢ SimplyConnectedSpace Y
    -/
    cases h; rw [simply_connected_iff_unique_homotopic]
    /-
      case intro
      Y : Type u_1
      inst✝ : TopologicalSpace Y
      left✝ : PathConnectedSpace Y
      right✝ : ∀ (x y : Y), Subsingleton (Path.Homotopic.Quotient x y)
      ⊢ And (Nonempty Y) (∀ (x y : Y), Nonempty (Unique (Path.Homotopic.Quotient x y …
    -/
    exact ⟨inferInstance, fun x y => ⟨uniqueOfSubsingleton ⟦PathConnectedSpace.somePath x y⟧⟩⟩⟩
    /-
      🎉 no goals
    -/


/-- Another version of `simply_connected_iff_paths_homotopic` -/
theorem simply_connected_iff_paths_homotopic' {Y : Type*} [TopologicalSpace Y] :
    SimplyConnectedSpace Y ↔
      PathConnectedSpace Y ∧ ∀ {x y : Y} (p₁ p₂ : Path x y), Path.Homotopic p₁ p₂ := by
  /-
    Y : Type u_1
    inst✝ : TopologicalSpace Y
    ⊢ Iff (SimplyConnectedSpace Y) (And (PathConnectedSpace Y) (∀ {x y : Y} (p₁ p₂ …
  -/
  convert simply_connected_iff_paths_homotopic (Y := Y)
  /-
    case h.e'_2.h.e'_2.h.h.a
    Y : Type u_1
    inst✝ : TopologicalSpace Y
    a✝¹ a✝ : Y
    ⊢ Iff (∀ (p₁ p₂ : Path a✝¹ a✝), p₁.Homotopic p₂) (Subsingleton (Path.Homotopic …
  -/
  simp [Path.Homotopic.Quotient, Setoid.eq_top_iff]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/

