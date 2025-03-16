/-- The topology coinduced by all maps from ℝⁿ into a space. -/
def TopologicalSpace.deltaGenerated (X : Type*) [TopologicalSpace X] : TopologicalSpace X :=
  ⨆ f : (n : ℕ) × C(((Fin n) → ℝ), X), coinduced f.2 inferInstance


/-- The delta-generated topology is also coinduced by a single map out of a sigma type. -/
lemma deltaGenerated_eq_coinduced : deltaGenerated X = coinduced
    (fun x : (f : (n : ℕ) × C(Fin n → ℝ, X)) × (Fin f.1 → ℝ) ↦ x.1.2 x.2) inferInstance := by
  /-
    X : Type u_1
    tX : TopologicalSpace X
    ⊢ Eq (TopologicalSpace.deltaGenerated X) (TopologicalSpace.coinduced (fun x => …
  -/
  rw [deltaGenerated, instTopologicalSpaceSigma, coinduced_iSup]; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- The delta-generated topology is at least as fine as the original one. -/
lemma deltaGenerated_le : deltaGenerated X ≤ tX :=
  iSup_le_iff.mpr fun f ↦ f.2.continuous.coinduced_le


/-- A set is open in `deltaGenerated X` iff all its preimages under continuous functions ℝⁿ → X are
  open. -/
lemma isOpen_deltaGenerated_iff {u : Set X} :
    IsOpen[deltaGenerated X] u ↔ ∀ n (p : C(Fin n → ℝ, X)), IsOpen (p ⁻¹' u) := by
  /-
    X : Type u_1
    tX : TopologicalSpace X
    u : Set X
    ⊢ Iff (IsOpen u) (∀ (n : Nat) (p : ContinuousMap (Fin n → Real) X), IsOpen (Se …
  -/
  simp_rw [deltaGenerated, isOpen_iSup_iff, isOpen_coinduced, Sigma.forall]
  /-
    🎉 no goals
  -/


/-- A map from ℝⁿ to X is continuous iff it is continuous regarding the
  delta-generated topology on X. Outside of this file, use the more general
  `continuous_to_deltaGenerated` instead. -/
private lemma continuous_euclidean_to_deltaGenerated {n : ℕ} {f : (Fin n → ℝ) → X} :
    Continuous[_, deltaGenerated X] f ↔ Continuous f := by
  /-
    X : Type u_1
    tX : TopologicalSpace X
    n : Nat
    f : (Fin n → Real) → X
    ⊢ Iff (Continuous f) (Continuous f)
  -/
  simp_rw [continuous_iff_coinduced_le]
  /-
    X : Type u_1
    tX : TopologicalSpace X
    n : Nat
    f : (Fin n → Real) → X
    ⊢ Iff (LE.le (TopologicalSpace.coinduced f Pi.topologicalSpace) (TopologicalSp …
  -/
  refine ⟨fun h ↦ h.trans deltaGenerated_le, fun h ↦ ?_⟩
  /-
    X : Type u_1
    tX : TopologicalSpace X
    n : Nat
    f : (Fin n → Real) → X
    h : LE.le (TopologicalSpace.coinduced f Pi.topologicalSpace) tX
    ⊢ LE.le (TopologicalSpace.coinduced f Pi.topologicalSpace) (TopologicalSpace.d …
  -/
  simp_rw [deltaGenerated]
  /-
    X : Type u_1
    tX : TopologicalSpace X
    n : Nat
    f : (Fin n → Real) → X
    h : LE.le (TopologicalSpace.coinduced f Pi.topologicalSpace) tX
    ⊢ LE.le (TopologicalSpace.coinduced f Pi.topologicalSpace) (iSup fun f => Topo …
  -/
  exact le_iSup_of_le (i := ⟨n, f, continuous_iff_coinduced_le.mpr h⟩) le_rfl
  /-
    🎉 no goals
  -/


/-- `deltaGenerated` is idempotent as a function `TopologicalSpace X → TopologicalSpace X`. -/
lemma deltaGenerated_deltaGenerated_eq :
    @deltaGenerated X (deltaGenerated X) = deltaGenerated X := by
  /-
    X : Type u_1
    tX : TopologicalSpace X
    ⊢ Eq (TopologicalSpace.deltaGenerated X) (TopologicalSpace.deltaGenerated X)
  -/
  ext u; simp_rw [isOpen_deltaGenerated_iff]; refine forall_congr' fun n ↦ ?_
  -- somewhat awkward because `ContinuousMap` doesn't play well with multiple topologies.
  /-
    case a.h.a
    X : Type u_1
    tX : TopologicalSpace X
    u : Set X
    n : Nat
    ⊢ Iff (∀ (p : ContinuousMap (Fin n → Real) X), IsOpen (Set.preimage (⇑p) u)) ( …
  -/
  refine ⟨fun h p ↦ h <| @ContinuousMap.mk _ _ _ (_) p ?_, fun h p ↦ h ⟨p, ?_⟩⟩
    /-
      case a.h.a.refine_1
      X : Type u_1
      tX : TopologicalSpace X
      u : Set X
      n : Nat
      h : ∀ (p : ContinuousMap (Fin n → Real) X), IsOpen (Set.preimage (⇑p) u)
      p : ContinuousMap (Fin n → Real) X
      ⊢ Continuous ⇑p
    -/
  · exact continuous_euclidean_to_deltaGenerated.mpr p.2
    /-
      🎉 no goals
    -/
    /-
      case a.h.a.refine_2
      X : Type u_1
      tX : TopologicalSpace X
      u : Set X
      n : Nat
      h : ∀ (p : ContinuousMap (Fin n → Real) X), IsOpen (Set.preimage (⇑p) u)
      p : ContinuousMap (Fin n → Real) X
      ⊢ Continuous ⇑p
    -/
  · exact continuous_euclidean_to_deltaGenerated.mp <| @ContinuousMap.continuous_toFun _ _ _ (_) p
    /-
      🎉 no goals
    -/


/-- A space is delta-generated if its topology is equal to the delta-generated topology, i.e.
  coinduced by all continuous maps ℝⁿ → X. Since the delta-generated topology is always finer
  than the original one, it suffices to show that it is also coarser. -/
class DeltaGeneratedSpace (X : Type*) [t : TopologicalSpace X] : Prop where
  le_deltaGenerated : t ≤ deltaGenerated X


lemma eq_deltaGenerated [DeltaGeneratedSpace X] : tX = deltaGenerated X :=
  eq_of_le_of_le DeltaGeneratedSpace.le_deltaGenerated deltaGenerated_le


/-- A subset of a delta-generated space is open iff its preimage is open for every
  continuous map from ℝⁿ to X. -/
lemma DeltaGeneratedSpace.isOpen_iff [DeltaGeneratedSpace X] {u : Set X} :
    IsOpen u ↔ ∀ (n : ℕ) (p : ContinuousMap ((Fin n) → ℝ) X), IsOpen (p ⁻¹' u) := by
  /-
    X : Type u_1
    tX : TopologicalSpace X
    inst✝ : DeltaGeneratedSpace X
    u : Set X
    ⊢ Iff (IsOpen u) (∀ (n : Nat) (p : ContinuousMap (Fin n → Real) X), IsOpen (Se …
  -/
  nth_rewrite 1 [eq_deltaGenerated (X := X)]; exact isOpen_deltaGenerated_iff
                                              /-
                                                🎉 no goals
                                              -/


/-- A map out of a delta-generated space is continuous iff it preserves continuity of maps
  from ℝⁿ into X. -/
lemma DeltaGeneratedSpace.continuous_iff [DeltaGeneratedSpace X] {f : X → Y} :
    Continuous f ↔ ∀ (n : ℕ) (p : C(((Fin n) → ℝ), X)), Continuous (f ∘ p) := by
  /-
    X : Type u_1
    Y : Type u_2
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : DeltaGeneratedSpace X
    f : X → Y
    ⊢ Iff (Continuous f) (∀ (n : Nat) (p : ContinuousMap (Fin n → Real) X), Contin …
  -/
  simp_rw [continuous_iff_coinduced_le]
  /-
    X : Type u_1
    Y : Type u_2
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : DeltaGeneratedSpace X
    f : X → Y
    ⊢ Iff (LE.le (TopologicalSpace.coinduced f tX) tY) (∀ (n : Nat) (p : Continuou …
  -/
  nth_rewrite 1 [eq_deltaGenerated (X := X), deltaGenerated]
  /-
    X : Type u_1
    Y : Type u_2
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : DeltaGeneratedSpace X
    f : X → Y
    ⊢ Iff (LE.le (TopologicalSpace.coinduced f (iSup fun f => TopologicalSpace.coi …
  -/
  simp [coinduced_compose]
  /-
    X : Type u_1
    Y : Type u_2
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : DeltaGeneratedSpace X
    f : X → Y
    ⊢ Iff (∀ (i : Sigma fun n => ContinuousMap (Fin n → Real) X), LE.le (Topologic …
  -/
  constructor
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      inst✝ : DeltaGeneratedSpace X
      f : X → Y
      ⊢ (∀ (i : Sigma fun n => ContinuousMap (Fin n → Real) X), LE.le (TopologicalSp …
    -/
  · intro h n p; apply h ⟨n, p⟩
                 /-
                   🎉 no goals
                 -/
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      inst✝ : DeltaGeneratedSpace X
      f : X → Y
      ⊢ (∀ (n : Nat) (p : ContinuousMap (Fin n → Real) X), LE.le (TopologicalSpace.c …
    -/
  · rintro h ⟨n, p⟩; apply h n p
                     /-
                       🎉 no goals
                     -/


/-- A map out of a delta-generated space is continuous iff it is continuous with respect
  to the delta-generated topology on the codomain. -/
lemma continuous_to_deltaGenerated [DeltaGeneratedSpace X] {f : X → Y} :
    Continuous[_, deltaGenerated Y] f ↔ Continuous f := by
  /-
    X : Type u_1
    Y : Type u_2
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : DeltaGeneratedSpace X
    f : X → Y
    ⊢ Iff (Continuous f) (Continuous f)
  -/
  simp_rw [DeltaGeneratedSpace.continuous_iff, continuous_euclidean_to_deltaGenerated]
  /-
    🎉 no goals
  -/


/-- The delta-generated topology on `X` does in fact turn `X` into a delta-generated space. -/
lemma deltaGeneratedSpace_deltaGenerated {X : Type*} {t : TopologicalSpace X} :
    @DeltaGeneratedSpace X (@deltaGenerated X t) := by
  /-
    X : Type u_3
    t : TopologicalSpace X
    ⊢ DeltaGeneratedSpace X
  -/
  let _ := @deltaGenerated X t; constructor; rw [@deltaGenerated_deltaGenerated_eq X t]
                                             /-
                                               🎉 no goals
                                             -/


lemma deltaGenerated_mono {X : Type*} {t₁ t₂ : TopologicalSpace X} (h : t₁ ≤ t₂) :
    @deltaGenerated X t₁ ≤ @deltaGenerated X t₂ := by
  rw [← continuous_id_iff_le, @continuous_to_deltaGenerated _ _
    (@deltaGenerated X t₁) t₂ deltaGeneratedSpace_deltaGenerated id]
  /-
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    h : LE.le t₁ t₂
    ⊢ Continuous id
  -/
  exact continuous_id_iff_le.2 <| (@deltaGenerated_le X t₁).trans h
  /-
    🎉 no goals
  -/


/-- Type synonym to be equipped with the delta-generated topology. -/
def of (X : Type*) := X


instance : TopologicalSpace (of X) := deltaGenerated X


instance : DeltaGeneratedSpace (of X) :=
  deltaGeneratedSpace_deltaGenerated


/-- The natural map from `DeltaGeneratedSpace.of X` to `X`. -/
def counit : (of X) → X := id


lemma continuous_counit : Continuous (counit : _ → X) := by
  /-
    X : Type u_1
    tX : TopologicalSpace X
    ⊢ Continuous DeltaGeneratedSpace.counit
  -/
  rw [continuous_iff_coinduced_le]; exact deltaGenerated_le
                                    /-
                                      🎉 no goals
                                    -/


/-- Delta-generated spaces are locally path-connected. -/
instance [DeltaGeneratedSpace X] : LocPathConnectedSpace X := by
  /-
    X : Type u_1
    Y : Type u_2
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : DeltaGeneratedSpace X
    ⊢ LocPathConnectedSpace X
  -/
  rw [eq_deltaGenerated (X := X), deltaGenerated_eq_coinduced]
  /-
    X : Type u_1
    Y : Type u_2
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : DeltaGeneratedSpace X
    ⊢ LocPathConnectedSpace X
  -/
  exact LocPathConnectedSpace.coinduced _
  /-
    🎉 no goals
  -/


/-- Delta-generated spaces are sequential. -/
instance [DeltaGeneratedSpace X] : SequentialSpace X := by
  /-
    X : Type u_1
    Y : Type u_2
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : DeltaGeneratedSpace X
    ⊢ SequentialSpace X
  -/
  rw [eq_deltaGenerated (X := X)]
  /-
    X : Type u_1
    Y : Type u_2
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : DeltaGeneratedSpace X
    ⊢ SequentialSpace X
  -/
  exact SequentialSpace.iSup fun p ↦ SequentialSpace.coinduced p.2
  /-
    🎉 no goals
  -/


omit tY in
/-- Any topology coinduced by a delta-generated topology is delta-generated. -/
lemma DeltaGeneratedSpace.coinduced [DeltaGeneratedSpace X] (f : X → Y) :
    @DeltaGeneratedSpace Y (tX.coinduced f) :=
  let _ := tX.coinduced f
  ⟨(continuous_to_deltaGenerated.2 continuous_coinduced_rng).coinduced_le⟩


/-- Suprema of delta-generated topologies are delta-generated. -/
protected lemma DeltaGeneratedSpace.iSup {X : Type*} {ι : Sort*} {t : ι → TopologicalSpace X}
    (h : ∀ i, @DeltaGeneratedSpace X (t i)) : @DeltaGeneratedSpace X (⨆ i, t i) :=
  let _ := ⨆ i, t i
  ⟨iSup_le_iff.2 fun i ↦ (h i).le_deltaGenerated.trans <| deltaGenerated_mono <| le_iSup t i⟩


/-- Suprema of delta-generated topologies are delta-generated. -/
protected lemma DeltaGeneratedSpace.sup {X : Type*} {t₁ t₂ : TopologicalSpace X}
    (h₁ : @DeltaGeneratedSpace X t₁) (h₂ : @DeltaGeneratedSpace X t₂) :
    @DeltaGeneratedSpace X (t₁ ⊔ t₂) := by
  /-
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    h₁ : DeltaGeneratedSpace X
    h₂ : DeltaGeneratedSpace X
    ⊢ DeltaGeneratedSpace X
  -/
  rw [sup_eq_iSup]
  /-
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    h₁ : DeltaGeneratedSpace X
    h₂ : DeltaGeneratedSpace X
    ⊢ DeltaGeneratedSpace X
  -/
  exact .iSup <| Bool.forall_bool.2 ⟨h₂, h₁⟩
  /-
    🎉 no goals
  -/


/-- Quotients of delta-generated spaces are delta-generated. -/
lemma Topology.IsQuotientMap.deltaGeneratedSpace [DeltaGeneratedSpace X]
    {f : X → Y} (h : IsQuotientMap f) : DeltaGeneratedSpace Y :=
  h.2 ▸ DeltaGeneratedSpace.coinduced f


/-- Quotients of delta-generated spaces are delta-generated. -/
instance Quot.deltaGeneratedSpace [DeltaGeneratedSpace X] {r : X → X → Prop} :
    DeltaGeneratedSpace (Quot r) :=
  isQuotientMap_quot_mk.deltaGeneratedSpace


/-- Quotients of delta-generated spaces are delta-generated. -/
instance Quotient.deltaGeneratedSpace [DeltaGeneratedSpace X] {s : Setoid X} :
    DeltaGeneratedSpace (Quotient s) :=
  isQuotientMap_quotient_mk'.deltaGeneratedSpace


/-- Disjoint unions of delta-generated spaces are delta-generated. -/
instance Sum.deltaGeneratedSpace [DeltaGeneratedSpace X] [DeltaGeneratedSpace Y] :
    DeltaGeneratedSpace (X ⊕ Y) :=
  DeltaGeneratedSpace.sup (.coinduced Sum.inl) (.coinduced Sum.inr)


/-- Disjoint unions of delta-generated spaces are delta-generated. -/
instance Sigma.deltaGeneratedSpace {ι : Type*} {X : ι → Type*} [∀ i, TopologicalSpace (X i)]
    [∀ i, DeltaGeneratedSpace (X i)] : DeltaGeneratedSpace (Σ i, X i) :=
  .iSup fun _ ↦ .coinduced _

