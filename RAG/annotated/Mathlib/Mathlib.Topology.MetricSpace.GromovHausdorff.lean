local notation "ℓ_infty_ℝ" => lp (fun n : ℕ => ℝ) ∞


/-- Equivalence relation identifying two nonempty compact sets which are isometric -/
private def IsometryRel (x : NonemptyCompacts ℓ_infty_ℝ) (y : NonemptyCompacts ℓ_infty_ℝ) : Prop :=
  Nonempty (x ≃ᵢ y)


/-- This is indeed an equivalence relation -/
private theorem equivalence_isometryRel : Equivalence IsometryRel :=
  ⟨fun _ => Nonempty.intro (IsometryEquiv.refl _), fun ⟨e⟩ => ⟨e.symm⟩, fun ⟨e⟩ ⟨f⟩ => ⟨e.trans f⟩⟩


/-- setoid instance identifying two isometric nonempty compact subspaces of ℓ^∞(ℝ) -/
instance IsometryRel.setoid : Setoid (NonemptyCompacts ℓ_infty_ℝ) :=
  Setoid.mk IsometryRel equivalence_isometryRel


/-- The Gromov-Hausdorff space -/
def GHSpace : Type :=
  Quotient IsometryRel.setoid


/-- Map any nonempty compact type to `GHSpace` -/
def toGHSpace (X : Type u) [MetricSpace X] [CompactSpace X] [Nonempty X] : GHSpace :=
  ⟦NonemptyCompacts.kuratowskiEmbedding X⟧


instance : Inhabited GHSpace :=
  ⟨Quot.mk _ ⟨⟨{0}, isCompact_singleton⟩, singleton_nonempty _⟩⟩


/-- A metric space representative of any abstract point in `GHSpace` -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not yet ported; removed @[nolint has_nonempty_instance]; why?
def GHSpace.Rep (p : GHSpace) : Type :=
  (Quotient.out p : NonemptyCompacts ℓ_infty_ℝ)


theorem eq_toGHSpace_iff {X : Type u} [MetricSpace X] [CompactSpace X] [Nonempty X]
    {p : NonemptyCompacts ℓ_infty_ℝ} :
    ⟦p⟧ = toGHSpace X ↔ ∃ Ψ : X → ℓ_infty_ℝ, Isometry Ψ ∧ range Ψ = p := by
  /-
    X : Type u
    inst✝² : MetricSpace X
    inst✝¹ : CompactSpace X
    inst✝ : Nonempty X
    p : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
    ⊢ Iff (Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid p) (GromovHausdorff. …
  -/
  simp only [toGHSpace, Quotient.eq]
  /-
    X : Type u
    inst✝² : MetricSpace X
    inst✝¹ : CompactSpace X
    inst✝ : Nonempty X
    p : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
    ⊢ Iff (GromovHausdorff.IsometryRel.setoid p (NonemptyCompacts.kuratowskiEmbedd …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      X : Type u
      inst✝² : MetricSpace X
      inst✝¹ : CompactSpace X
      inst✝ : Nonempty X
      p : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      h : GromovHausdorff.IsometryRel.setoid p (NonemptyCompacts.kuratowskiEmbedding …
      ⊢ Exists fun Ψ => And (Isometry Ψ) (Eq (Set.range Ψ) ↑p)
    -/
  · rcases Setoid.symm h with ⟨e⟩
    /-
      case refine_1.intro
      X : Type u
      inst✝² : MetricSpace X
      inst✝¹ : CompactSpace X
      inst✝ : Nonempty X
      p : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      h : GromovHausdorff.IsometryRel.setoid p (NonemptyCompacts.kuratowskiEmbedding …
      e : IsometryEquiv (Subtype fun x => Membership.mem (NonemptyCompacts.kuratowsk …
      ⊢ Exists fun Ψ => And (Isometry Ψ) (Eq (Set.range Ψ) ↑p)
    -/
    have f := (kuratowskiEmbedding.isometry X).isometryEquivOnRange.trans e
    /-
      case refine_1.intro
      X : Type u
      inst✝² : MetricSpace X
      inst✝¹ : CompactSpace X
      inst✝ : Nonempty X
      p : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      h : GromovHausdorff.IsometryRel.setoid p (NonemptyCompacts.kuratowskiEmbedding …
      e : IsometryEquiv (Subtype fun x => Membership.mem (NonemptyCompacts.kuratowsk …
      f : IsometryEquiv X (Subtype fun x => Membership.mem p x)
      ⊢ Exists fun Ψ => And (Isometry Ψ) (Eq (Set.range Ψ) ↑p)
    -/
    use fun x => f x, isometry_subtype_coe.comp f.isometry
    /-
      case right
      X : Type u
      inst✝² : MetricSpace X
      inst✝¹ : CompactSpace X
      inst✝ : Nonempty X
      p : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      h : GromovHausdorff.IsometryRel.setoid p (NonemptyCompacts.kuratowskiEmbedding …
      e : IsometryEquiv (Subtype fun x => Membership.mem (NonemptyCompacts.kuratowsk …
      f : IsometryEquiv X (Subtype fun x => Membership.mem p x)
      ⊢ Eq (Set.range fun x => ↑(f x)) ↑p
    -/
    erw [range_comp, f.range_eq_univ, Set.image_univ, Subtype.range_coe]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u
      inst✝² : MetricSpace X
      inst✝¹ : CompactSpace X
      inst✝ : Nonempty X
      p : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      ⊢ (Exists fun Ψ => And (Isometry Ψ) (Eq (Set.range Ψ) ↑p)) → GromovHausdorff.I …
    -/
  · rintro ⟨Ψ, ⟨isomΨ, rangeΨ⟩⟩
    have f :=
      ((kuratowskiEmbedding.isometry X).isometryEquivOnRange.symm.trans
          isomΨ.isometryEquivOnRange).symm
    have E : (range Ψ ≃ᵢ NonemptyCompacts.kuratowskiEmbedding X)
        = (p ≃ᵢ range (kuratowskiEmbedding X)) := by
      dsimp only [NonemptyCompacts.kuratowskiEmbedding]; rw [rangeΨ]; rfl
    /-
      case refine_2.intro.intro
      X : Type u
      inst✝² : MetricSpace X
      inst✝¹ : CompactSpace X
      inst✝ : Nonempty X
      p : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      Ψ : X → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      isomΨ : Isometry Ψ
      rangeΨ : Eq (Set.range Ψ) ↑p
      f : IsometryEquiv ↑(Set.range Ψ) ↑(Set.range (kuratowskiEmbedding X))
      E : Eq (IsometryEquiv (↑(Set.range Ψ)) (Subtype fun x => Membership.mem (Nonem …
      ⊢ GromovHausdorff.IsometryRel.setoid p (NonemptyCompacts.kuratowskiEmbedding X)
    -/
    exact ⟨cast E f⟩
    /-
      🎉 no goals
    -/


theorem eq_toGHSpace {p : NonemptyCompacts ℓ_infty_ℝ} : ⟦p⟧ = toGHSpace p :=
  eq_toGHSpace_iff.2 ⟨fun x => x, isometry_subtype_coe, Subtype.range_coe⟩


instance repGHSpaceMetricSpace {p : GHSpace} : MetricSpace p.Rep :=
  inferInstanceAs <| MetricSpace p.out


instance rep_gHSpace_compactSpace {p : GHSpace} : CompactSpace p.Rep :=
  inferInstanceAs <| CompactSpace p.out


instance rep_gHSpace_nonempty {p : GHSpace} : Nonempty p.Rep :=
  inferInstanceAs <| Nonempty p.out


theorem GHSpace.toGHSpace_rep (p : GHSpace) : toGHSpace p.Rep = p := by
  /-
    p : GromovHausdorff.GHSpace
    ⊢ Eq (GromovHausdorff.toGHSpace p.Rep) p
  -/
  change toGHSpace (Quot.out p : NonemptyCompacts ℓ_infty_ℝ) = p
  /-
    p : GromovHausdorff.GHSpace
    ⊢ Eq (GromovHausdorff.toGHSpace (Subtype fun x => Membership.mem (Quot.out p)  …
  -/
  rw [← eq_toGHSpace]
  /-
    p : GromovHausdorff.GHSpace
    ⊢ Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid (Quot.out p)) p
  -/
  exact Quot.out_eq p
  /-
    🎉 no goals
  -/


/-- Two nonempty compact spaces have the same image in `GHSpace` if and only if they are
isometric. -/
theorem toGHSpace_eq_toGHSpace_iff_isometryEquiv {X : Type u} [MetricSpace X] [CompactSpace X]
    [Nonempty X] {Y : Type v} [MetricSpace Y] [CompactSpace Y] [Nonempty Y] :
    toGHSpace X = toGHSpace Y ↔ Nonempty (X ≃ᵢ Y) :=
  ⟨by
    /-
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      ⊢ Eq (GromovHausdorff.toGHSpace X) (GromovHausdorff.toGHSpace Y) → Nonempty (I …
    -/
    simp only [toGHSpace]
    /-
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      ⊢ Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid (NonemptyCompacts.kuratow …
    -/
    rw [Quotient.eq]
    /-
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      ⊢ GromovHausdorff.IsometryRel.setoid (NonemptyCompacts.kuratowskiEmbedding X)  …
    -/
    rintro ⟨e⟩
    have I :
      (NonemptyCompacts.kuratowskiEmbedding X ≃ᵢ NonemptyCompacts.kuratowskiEmbedding Y) =
        (range (kuratowskiEmbedding X) ≃ᵢ range (kuratowskiEmbedding Y)) := by
      dsimp only [NonemptyCompacts.kuratowskiEmbedding]; rfl
    /-
      case intro
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      e : IsometryEquiv (Subtype fun x => Membership.mem (NonemptyCompacts.kuratowsk …
      I : Eq (IsometryEquiv (Subtype fun x => Membership.mem (NonemptyCompacts.kurat …
      ⊢ Nonempty (IsometryEquiv X Y)
    -/
    have f := (kuratowskiEmbedding.isometry X).isometryEquivOnRange
    /-
      case intro
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      e : IsometryEquiv (Subtype fun x => Membership.mem (NonemptyCompacts.kuratowsk …
      I : Eq (IsometryEquiv (Subtype fun x => Membership.mem (NonemptyCompacts.kurat …
      f : IsometryEquiv X ↑(Set.range (kuratowskiEmbedding X))
      ⊢ Nonempty (IsometryEquiv X Y)
    -/
    have g := (kuratowskiEmbedding.isometry Y).isometryEquivOnRange.symm
    /-
      case intro
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      e : IsometryEquiv (Subtype fun x => Membership.mem (NonemptyCompacts.kuratowsk …
      I : Eq (IsometryEquiv (Subtype fun x => Membership.mem (NonemptyCompacts.kurat …
      f : IsometryEquiv X ↑(Set.range (kuratowskiEmbedding X))
      g : IsometryEquiv (↑(Set.range (kuratowskiEmbedding Y))) Y
      ⊢ Nonempty (IsometryEquiv X Y)
    -/
    exact ⟨f.trans <| (cast I e).trans g⟩, by
    /-
      🎉 no goals
    -/
    /-
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      ⊢ Nonempty (IsometryEquiv X Y) → Eq (GromovHausdorff.toGHSpace X) (GromovHausd …
    -/
    rintro ⟨e⟩
    /-
      case intro
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      e : IsometryEquiv X Y
      ⊢ Eq (GromovHausdorff.toGHSpace X) (GromovHausdorff.toGHSpace Y)
    -/
    simp only [toGHSpace, Quotient.eq']
    /-
      case intro
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      e : IsometryEquiv X Y
      ⊢ Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid (NonemptyCompacts.kuratow …
    -/
    have f := (kuratowskiEmbedding.isometry X).isometryEquivOnRange.symm
    /-
      case intro
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      e : IsometryEquiv X Y
      f : IsometryEquiv (↑(Set.range (kuratowskiEmbedding X))) X
      ⊢ Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid (NonemptyCompacts.kuratow …
    -/
    have g := (kuratowskiEmbedding.isometry Y).isometryEquivOnRange
    have I :
      (range (kuratowskiEmbedding X) ≃ᵢ range (kuratowskiEmbedding Y)) =
        (NonemptyCompacts.kuratowskiEmbedding X ≃ᵢ NonemptyCompacts.kuratowskiEmbedding Y) := by
      dsimp only [NonemptyCompacts.kuratowskiEmbedding]; rfl
    /-
      case intro
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      e : IsometryEquiv X Y
      f : IsometryEquiv (↑(Set.range (kuratowskiEmbedding X))) X
      g : IsometryEquiv Y ↑(Set.range (kuratowskiEmbedding Y))
      I : Eq (IsometryEquiv ↑(Set.range (kuratowskiEmbedding X)) ↑(Set.range (kurato …
      ⊢ Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid (NonemptyCompacts.kuratow …
    -/
    rw [Quotient.eq]
    /-
      case intro
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      e : IsometryEquiv X Y
      f : IsometryEquiv (↑(Set.range (kuratowskiEmbedding X))) X
      g : IsometryEquiv Y ↑(Set.range (kuratowskiEmbedding Y))
      I : Eq (IsometryEquiv ↑(Set.range (kuratowskiEmbedding X)) ↑(Set.range (kurato …
      ⊢ GromovHausdorff.IsometryRel.setoid (NonemptyCompacts.kuratowskiEmbedding X)  …
    -/
    exact ⟨cast I ((f.trans e).trans g)⟩⟩
    /-
      🎉 no goals
    -/


/-- Distance on `GHSpace`: the distance between two nonempty compact spaces is the infimum
Hausdorff distance between isometric copies of the two spaces in a metric space. For the definition,
we only consider embeddings in `ℓ^∞(ℝ)`, but we will prove below that it works for all spaces. -/
instance : Dist GHSpace where
  dist x y := sInf <| (fun p : NonemptyCompacts ℓ_infty_ℝ × NonemptyCompacts ℓ_infty_ℝ =>
    hausdorffDist (p.1 : Set ℓ_infty_ℝ) p.2) '' { a | ⟦a⟧ = x } ×ˢ { b | ⟦b⟧ = y }


/-- The Gromov-Hausdorff distance between two nonempty compact metric spaces, equal by definition to
the distance of the equivalence classes of these spaces in the Gromov-Hausdorff space. -/
def ghDist (X : Type u) (Y : Type v) [MetricSpace X] [Nonempty X] [CompactSpace X] [MetricSpace Y]
    [Nonempty Y] [CompactSpace Y] : ℝ :=
  dist (toGHSpace X) (toGHSpace Y)


theorem dist_ghDist (p q : GHSpace) : dist p q = ghDist p.Rep q.Rep := by
  /-
    p q : GromovHausdorff.GHSpace
    ⊢ Eq (Dist.dist p q) (GromovHausdorff.ghDist p.Rep q.Rep)
  -/
  rw [ghDist, p.toGHSpace_rep, q.toGHSpace_rep]
  /-
    🎉 no goals
  -/


/-- The Gromov-Hausdorff distance between two spaces is bounded by the Hausdorff distance
of isometric copies of the spaces, in any metric space. -/
theorem ghDist_le_hausdorffDist {X : Type u} [MetricSpace X] [CompactSpace X] [Nonempty X]
    {Y : Type v} [MetricSpace Y] [CompactSpace Y] [Nonempty Y] {γ : Type w} [MetricSpace γ]
    {Φ : X → γ} {Ψ : Y → γ} (ha : Isometry Φ) (hb : Isometry Ψ) :
    ghDist X Y ≤ hausdorffDist (range Φ) (range Ψ) := by
  /- For the proof, we want to embed `γ` in `ℓ^∞(ℝ)`, to say that the Hausdorff distance is realized
    in `ℓ^∞(ℝ)` and therefore bounded below by the Gromov-Hausdorff-distance. However, `γ` is not
    separable in general. We restrict to the union of the images of `X` and `Y` in `γ`, which is
    separable and therefore embeddable in `ℓ^∞(ℝ)`. -/
  /-
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  rcases exists_mem_of_nonempty X with ⟨xX, _⟩
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  let s : Set γ := range Φ ∪ range Ψ
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  let Φ' : X → Subtype s := fun y => ⟨Φ y, mem_union_left _ (mem_range_self _)⟩
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  let Ψ' : Y → Subtype s := fun y => ⟨Ψ y, mem_union_right _ (mem_range_self _)⟩
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  have IΦ' : Isometry Φ' := fun x y => ha x y
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  have IΨ' : Isometry Ψ' := fun x y => hb x y
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  have : IsCompact s := (isCompact_range ha.continuous).union (isCompact_range hb.continuous)
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    this : IsCompact s
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  letI : MetricSpace (Subtype s) := by infer_instance
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    this✝ : IsCompact s
    this : MetricSpace (Subtype s) := inferInstance
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  haveI : CompactSpace (Subtype s) := ⟨isCompact_iff_isCompact_univ.1 ‹IsCompact s›⟩
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    this✝¹ : IsCompact s
    this✝ : MetricSpace (Subtype s) := inferInstance
    this : CompactSpace (Subtype s)
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  haveI : Nonempty (Subtype s) := ⟨Φ' xX⟩
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    this✝² : IsCompact s
    this✝¹ : MetricSpace (Subtype s) := inferInstance
    this✝ : CompactSpace (Subtype s)
    this : Nonempty (Subtype s)
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  have ΦΦ' : Φ = Subtype.val ∘ Φ' := by funext; rfl
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    this✝² : IsCompact s
    this✝¹ : MetricSpace (Subtype s) := inferInstance
    this✝ : CompactSpace (Subtype s)
    this : Nonempty (Subtype s)
    ΦΦ' : Eq Φ (Function.comp Subtype.val Φ')
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  have ΨΨ' : Ψ = Subtype.val ∘ Ψ' := by funext; rfl
  have : hausdorffDist (range Φ) (range Ψ) = hausdorffDist (range Φ') (range Ψ') := by
    rw [ΦΦ', ΨΨ', range_comp, range_comp]
    exact hausdorffDist_image isometry_subtype_coe
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    this✝³ : IsCompact s
    this✝² : MetricSpace (Subtype s) := inferInstance
    this✝¹ : CompactSpace (Subtype s)
    this✝ : Nonempty (Subtype s)
    ΦΦ' : Eq Φ (Function.comp Subtype.val Φ')
    ΨΨ' : Eq Ψ (Function.comp Subtype.val Ψ')
    this : Eq (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ)) (Metric.hausdorff …
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ) (Set. …
  -/
  rw [this]
  -- Embed `s` in `ℓ^∞(ℝ)` through its Kuratowski embedding
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    this✝³ : IsCompact s
    this✝² : MetricSpace (Subtype s) := inferInstance
    this✝¹ : CompactSpace (Subtype s)
    this✝ : Nonempty (Subtype s)
    ΦΦ' : Eq Φ (Function.comp Subtype.val Φ')
    ΨΨ' : Eq Ψ (Function.comp Subtype.val Ψ')
    this : Eq (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ)) (Metric.hausdorff …
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ') (Set …
  -/
  let F := kuratowskiEmbedding (Subtype s)
  have : hausdorffDist (F '' range Φ') (F '' range Ψ') = hausdorffDist (range Φ') (range Ψ') :=
    hausdorffDist_image (kuratowskiEmbedding.isometry _)
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    this✝⁴ : IsCompact s
    this✝³ : MetricSpace (Subtype s) := inferInstance
    this✝² : CompactSpace (Subtype s)
    this✝¹ : Nonempty (Subtype s)
    ΦΦ' : Eq Φ (Function.comp Subtype.val Φ')
    ΨΨ' : Eq Ψ (Function.comp Subtype.val Ψ')
    this✝ : Eq (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ)) (Metric.hausdorf …
    F : Subtype s → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x …
    this : Eq (Metric.hausdorffDist (Set.image F (Set.range Φ')) (Set.image F (Set …
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range Φ') (Set …
  -/
  rw [← this]
  -- Let `A` and `B` be the images of `X` and `Y` under this embedding. They are in `ℓ^∞(ℝ)`, and
  -- their Hausdorff distance is the same as in the original space.
  let A : NonemptyCompacts ℓ_infty_ℝ :=
    ⟨⟨F '' range Φ',
        (isCompact_range IΦ'.continuous).image (kuratowskiEmbedding.isometry _).continuous⟩,
      (range_nonempty _).image _⟩
  let B : NonemptyCompacts ℓ_infty_ℝ :=
    ⟨⟨F '' range Ψ',
        (isCompact_range IΨ'.continuous).image (kuratowskiEmbedding.isometry _).continuous⟩,
      (range_nonempty _).image _⟩
  have AX : ⟦A⟧ = toGHSpace X := by
    rw [eq_toGHSpace_iff]
    exact ⟨fun x => F (Φ' x), (kuratowskiEmbedding.isometry _).comp IΦ', range_comp _ _⟩
  have BY : ⟦B⟧ = toGHSpace Y := by
    rw [eq_toGHSpace_iff]
    exact ⟨fun x => F (Ψ' x), (kuratowskiEmbedding.isometry _).comp IΨ', range_comp _ _⟩
  /-
    case intro
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    this✝⁴ : IsCompact s
    this✝³ : MetricSpace (Subtype s) := inferInstance
    this✝² : CompactSpace (Subtype s)
    this✝¹ : Nonempty (Subtype s)
    ΦΦ' : Eq Φ (Function.comp Subtype.val Φ')
    ΨΨ' : Eq Ψ (Function.comp Subtype.val Ψ')
    this✝ : Eq (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ)) (Metric.hausdorf …
    F : Subtype s → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x …
    this : Eq (Metric.hausdorffDist (Set.image F (Set.range Φ')) (Set.image F (Set …
    A : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
    B : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
    AX : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid A) (GromovHausdorff.to …
    BY : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid B) (GromovHausdorff.to …
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.image F (Set.r …
  -/
  refine csInf_le ⟨0, ?_⟩ ?_
  · simp only [lowerBounds, mem_image, mem_prod, mem_setOf_eq, Prod.exists, and_imp,
      forall_exists_index]
    /-
      case intro.refine_1
      X : Type u
      inst✝⁶ : MetricSpace X
      inst✝⁵ : CompactSpace X
      inst✝⁴ : Nonempty X
      Y : Type v
      inst✝³ : MetricSpace Y
      inst✝² : CompactSpace Y
      inst✝¹ : Nonempty Y
      γ : Type w
      inst✝ : MetricSpace γ
      Φ : X → γ
      Ψ : Y → γ
      ha : Isometry Φ
      hb : Isometry Ψ
      xX : X
      h✝ : Membership.mem Set.univ xX
      s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
      Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
      Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
      IΦ' : Isometry Φ'
      IΨ' : Isometry Ψ'
      this✝⁴ : IsCompact s
      this✝³ : MetricSpace (Subtype s) := inferInstance
      this✝² : CompactSpace (Subtype s)
      this✝¹ : Nonempty (Subtype s)
      ΦΦ' : Eq Φ (Function.comp Subtype.val Φ')
      ΨΨ' : Eq Ψ (Function.comp Subtype.val Ψ')
      this✝ : Eq (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ)) (Metric.hausdorf …
      F : Subtype s → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x …
      this : Eq (Metric.hausdorffDist (Set.image F (Set.range Φ')) (Set.image F (Set …
      A : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      B : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      AX : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid A) (GromovHausdorff.to …
      BY : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid B) (GromovHausdorff.to …
      ⊢ ∀ ⦃a : Real⦄ (x x_1 : TopologicalSpace.NonemptyCompacts (Subtype fun x => Me …
    -/
    intro t _ _ _ _ ht
    /-
      case intro.refine_1
      X : Type u
      inst✝⁶ : MetricSpace X
      inst✝⁵ : CompactSpace X
      inst✝⁴ : Nonempty X
      Y : Type v
      inst✝³ : MetricSpace Y
      inst✝² : CompactSpace Y
      inst✝¹ : Nonempty Y
      γ : Type w
      inst✝ : MetricSpace γ
      Φ : X → γ
      Ψ : Y → γ
      ha : Isometry Φ
      hb : Isometry Ψ
      xX : X
      h✝ : Membership.mem Set.univ xX
      s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
      Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
      Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
      IΦ' : Isometry Φ'
      IΨ' : Isometry Ψ'
      this✝⁴ : IsCompact s
      this✝³ : MetricSpace (Subtype s) := inferInstance
      this✝² : CompactSpace (Subtype s)
      this✝¹ : Nonempty (Subtype s)
      ΦΦ' : Eq Φ (Function.comp Subtype.val Φ')
      ΨΨ' : Eq Ψ (Function.comp Subtype.val Ψ')
      this✝ : Eq (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ)) (Metric.hausdorf …
      F : Subtype s → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x …
      this : Eq (Metric.hausdorffDist (Set.image F (Set.range Φ')) (Set.image F (Set …
      A : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      B : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      AX : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid A) (GromovHausdorff.to …
      BY : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid B) (GromovHausdorff.to …
      t : Real
      x✝¹ x✝ : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (l …
      a✝¹ : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid x✝¹) (GromovHausdorff …
      a✝ : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid x✝) (GromovHausdorff.t …
      ht : Eq (Metric.hausdorffDist ↑x✝¹ ↑x✝) t
      ⊢ LE.le 0 t
    -/
    rw [← ht]
    /-
      case intro.refine_1
      X : Type u
      inst✝⁶ : MetricSpace X
      inst✝⁵ : CompactSpace X
      inst✝⁴ : Nonempty X
      Y : Type v
      inst✝³ : MetricSpace Y
      inst✝² : CompactSpace Y
      inst✝¹ : Nonempty Y
      γ : Type w
      inst✝ : MetricSpace γ
      Φ : X → γ
      Ψ : Y → γ
      ha : Isometry Φ
      hb : Isometry Ψ
      xX : X
      h✝ : Membership.mem Set.univ xX
      s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
      Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
      Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
      IΦ' : Isometry Φ'
      IΨ' : Isometry Ψ'
      this✝⁴ : IsCompact s
      this✝³ : MetricSpace (Subtype s) := inferInstance
      this✝² : CompactSpace (Subtype s)
      this✝¹ : Nonempty (Subtype s)
      ΦΦ' : Eq Φ (Function.comp Subtype.val Φ')
      ΨΨ' : Eq Ψ (Function.comp Subtype.val Ψ')
      this✝ : Eq (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ)) (Metric.hausdorf …
      F : Subtype s → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x …
      this : Eq (Metric.hausdorffDist (Set.image F (Set.range Φ')) (Set.image F (Set …
      A : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      B : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      AX : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid A) (GromovHausdorff.to …
      BY : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid B) (GromovHausdorff.to …
      t : Real
      x✝¹ x✝ : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (l …
      a✝¹ : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid x✝¹) (GromovHausdorff …
      a✝ : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid x✝) (GromovHausdorff.t …
      ht : Eq (Metric.hausdorffDist ↑x✝¹ ↑x✝) t
      ⊢ LE.le 0 (Metric.hausdorffDist ↑x✝¹ ↑x✝)
    -/
    exact hausdorffDist_nonneg
    /-
      🎉 no goals
    -/
  /-
    case intro.refine_2
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    this✝⁴ : IsCompact s
    this✝³ : MetricSpace (Subtype s) := inferInstance
    this✝² : CompactSpace (Subtype s)
    this✝¹ : Nonempty (Subtype s)
    ΦΦ' : Eq Φ (Function.comp Subtype.val Φ')
    ΨΨ' : Eq Ψ (Function.comp Subtype.val Ψ')
    this✝ : Eq (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ)) (Metric.hausdorf …
    F : Subtype s → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x …
    this : Eq (Metric.hausdorffDist (Set.image F (Set.range Φ')) (Set.image F (Set …
    A : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
    B : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
    AX : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid A) (GromovHausdorff.to …
    BY : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid B) (GromovHausdorff.to …
    ⊢ Membership.mem (Set.image (fun p => Metric.hausdorffDist ↑p.1 ↑p.2) (SProd.s …
  -/
  apply (mem_image _ _ _).2
  /-
    case intro.refine_2
    X : Type u
    inst✝⁶ : MetricSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : Nonempty X
    Y : Type v
    inst✝³ : MetricSpace Y
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty Y
    γ : Type w
    inst✝ : MetricSpace γ
    Φ : X → γ
    Ψ : Y → γ
    ha : Isometry Φ
    hb : Isometry Ψ
    xX : X
    h✝ : Membership.mem Set.univ xX
    s : Set γ := Union.union (Set.range Φ) (Set.range Ψ)
    Φ' : X → Subtype s := fun y => ⟨Φ y, ⋯⟩
    Ψ' : Y → Subtype s := fun y => ⟨Ψ y, ⋯⟩
    IΦ' : Isometry Φ'
    IΨ' : Isometry Ψ'
    this✝⁴ : IsCompact s
    this✝³ : MetricSpace (Subtype s) := inferInstance
    this✝² : CompactSpace (Subtype s)
    this✝¹ : Nonempty (Subtype s)
    ΦΦ' : Eq Φ (Function.comp Subtype.val Φ')
    ΨΨ' : Eq Ψ (Function.comp Subtype.val Ψ')
    this✝ : Eq (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ)) (Metric.hausdorf …
    F : Subtype s → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x …
    this : Eq (Metric.hausdorffDist (Set.image F (Set.range Φ')) (Set.image F (Set …
    A : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
    B : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
    AX : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid A) (GromovHausdorff.to …
    BY : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid B) (GromovHausdorff.to …
    ⊢ Exists fun x => And (Membership.mem (SProd.sprod (setOf fun a => Eq (Quotien …
  -/
  exists (⟨A, B⟩ : NonemptyCompacts ℓ_infty_ℝ × NonemptyCompacts ℓ_infty_ℝ)
  /-
    🎉 no goals
  -/


/-- The optimal coupling constructed above realizes exactly the Gromov-Hausdorff distance,
essentially by design. -/
theorem hausdorffDist_optimal {X : Type u} [MetricSpace X] [CompactSpace X] [Nonempty X]
    {Y : Type v} [MetricSpace Y] [CompactSpace Y] [Nonempty Y] :
    hausdorffDist (range (optimalGHInjl X Y)) (range (optimalGHInjr X Y)) = ghDist X Y := by
  /-
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    ⊢ Eq (Metric.hausdorffDist (Set.range (GromovHausdorff.optimalGHInjl X Y)) (Se …
  -/
  inhabit X; inhabit Y
  /- we only need to check the inequality `≤`, as the other one follows from the previous lemma.
       As the Gromov-Hausdorff distance is an infimum, we need to check that the Hausdorff distance
       in the optimal coupling is smaller than the Hausdorff distance of any coupling.
       First, we check this for couplings which already have small Hausdorff distance: in this
       case, the induced "distance" on `X ⊕ Y` belongs to the candidates family introduced in the
       definition of the optimal coupling, and the conclusion follows from the optimality
       of the optimal coupling within this family.
    -/
  have A :
    ∀ p q : NonemptyCompacts ℓ_infty_ℝ,
      ⟦p⟧ = toGHSpace X →
        ⟦q⟧ = toGHSpace Y →
          hausdorffDist (p : Set ℓ_infty_ℝ) q < diam (univ : Set X) + 1 + diam (univ : Set Y) →
            hausdorffDist (range (optimalGHInjl X Y)) (range (optimalGHInjr X Y)) ≤
              hausdorffDist (p : Set ℓ_infty_ℝ) q := by
    intro p q hp hq bound
    rcases eq_toGHSpace_iff.1 hp with ⟨Φ, ⟨Φisom, Φrange⟩⟩
    rcases eq_toGHSpace_iff.1 hq with ⟨Ψ, ⟨Ψisom, Ψrange⟩⟩
    have I : diam (range Φ ∪ range Ψ) ≤ 2 * diam (univ : Set X) + 1 + 2 * diam (univ : Set Y) := by
      rcases exists_mem_of_nonempty X with ⟨xX, _⟩
      have : ∃ y ∈ range Ψ, dist (Φ xX) y < diam (univ : Set X) + 1 + diam (univ : Set Y) := by
        rw [Ψrange]
        have : Φ xX ∈ (p : Set _) := Φrange ▸ (mem_range_self _)
        exact
          exists_dist_lt_of_hausdorffDist_lt this bound
            (hausdorffEdist_ne_top_of_nonempty_of_bounded p.nonempty q.nonempty
              p.isCompact.isBounded q.isCompact.isBounded)
      rcases this with ⟨y, hy, dy⟩
      rcases mem_range.1 hy with ⟨z, hzy⟩
      rw [← hzy] at dy
      have DΦ : diam (range Φ) = diam (univ : Set X) := Φisom.diam_range
      have DΨ : diam (range Ψ) = diam (univ : Set Y) := Ψisom.diam_range
      calc
        diam (range Φ ∪ range Ψ) ≤ diam (range Φ) + dist (Φ xX) (Ψ z) + diam (range Ψ) :=
          diam_union (mem_range_self _) (mem_range_self _)
        _ ≤
            diam (univ : Set X) + (diam (univ : Set X) + 1 + diam (univ : Set Y)) +
              diam (univ : Set Y) := by
          rw [DΦ, DΨ]
          gcongr
          -- apply add_le_add (add_le_add le_rfl (le_of_lt dy)) le_rfl
        _ = 2 * diam (univ : Set X) + 1 + 2 * diam (univ : Set Y) := by ring
    let f : X ⊕ Y → ℓ_infty_ℝ := fun x =>
      match x with
      | inl y => Φ y
      | inr z => Ψ z
    let F : (X ⊕ Y) × (X ⊕ Y) → ℝ := fun p => dist (f p.1) (f p.2)
    -- check that the induced "distance" is a candidate
    have Fgood : F ∈ candidates X Y := by
      simp only [F, candidates, forall_const, add_comm, eq_self_iff_true,
        dist_eq_zero, and_self_iff, Set.mem_setOf_eq]
      repeat' constructor
      · exact fun x y =>
          calc
            F (inl x, inl y) = dist (Φ x) (Φ y) := rfl
            _ = dist x y := Φisom.dist_eq x y

      · exact fun x y =>
          calc
            F (inr x, inr y) = dist (Ψ x) (Ψ y) := rfl
            _ = dist x y := Ψisom.dist_eq x y

      · exact fun x y => dist_comm _ _
      · exact fun x y z => dist_triangle _ _ _
      · exact fun x y =>
          calc
            F (x, y) ≤ diam (range Φ ∪ range Ψ) := by
              have A : ∀ z : X ⊕ Y, f z ∈ range Φ ∪ range Ψ := by
                intro z
                cases z
                · apply mem_union_left; apply mem_range_self
                · apply mem_union_right; apply mem_range_self
              refine dist_le_diam_of_mem ?_ (A _) (A _)
              rw [Φrange, Ψrange]
              exact (p ⊔ q).isCompact.isBounded
            _ ≤ 2 * diam (univ : Set X) + 1 + 2 * diam (univ : Set Y) := I
    let Fb := candidatesBOfCandidates F Fgood
    have : hausdorffDist (range (optimalGHInjl X Y)) (range (optimalGHInjr X Y)) ≤ HD Fb :=
      hausdorffDist_optimal_le_HD _ _ (candidatesBOfCandidates_mem F Fgood)
    refine le_trans this (le_of_forall_le_of_dense fun r hr => ?_)
    have I1 : ∀ x : X, (⨅ y, Fb (inl x, inr y)) ≤ r := by
      intro x
      have : f (inl x) ∈ (p : Set _) := Φrange ▸ (mem_range_self _)
      rcases exists_dist_lt_of_hausdorffDist_lt this hr
          (hausdorffEdist_ne_top_of_nonempty_of_bounded p.nonempty q.nonempty p.isCompact.isBounded
            q.isCompact.isBounded) with
        ⟨z, zq, hz⟩
      have : z ∈ range Ψ := by rwa [← Ψrange] at zq
      rcases mem_range.1 this with ⟨y, hy⟩
      calc
        (⨅ y, Fb (inl x, inr y)) ≤ Fb (inl x, inr y) :=
          ciInf_le (by simpa only [add_zero] using HD_below_aux1 0) y
        _ = dist (Φ x) (Ψ y) := rfl
        _ = dist (f (inl x)) z := by rw [hy]
        _ ≤ r := le_of_lt hz

    have I2 : ∀ y : Y, (⨅ x, Fb (inl x, inr y)) ≤ r := by
      intro y
      have : f (inr y) ∈ (q : Set _) := Ψrange ▸ (mem_range_self _)
      rcases exists_dist_lt_of_hausdorffDist_lt' this hr
          (hausdorffEdist_ne_top_of_nonempty_of_bounded p.nonempty q.nonempty p.isCompact.isBounded
            q.isCompact.isBounded) with
        ⟨z, zq, hz⟩
      have : z ∈ range Φ := by rwa [← Φrange] at zq
      rcases mem_range.1 this with ⟨x, hx⟩
      calc
        (⨅ x, Fb (inl x, inr y)) ≤ Fb (inl x, inr y) :=
          ciInf_le (by simpa only [add_zero] using HD_below_aux2 0) x
        _ = dist (Φ x) (Ψ y) := rfl
        _ = dist z (f (inr y)) := by rw [hx]
        _ ≤ r := le_of_lt hz

    simp only [HD, ciSup_le I1, ciSup_le I2, max_le_iff, and_self_iff]
  /- Get the same inequality for any coupling. If the coupling is quite good, the desired
    inequality has been proved above. If it is bad, then the inequality is obvious. -/
  have B :
    ∀ p q : NonemptyCompacts ℓ_infty_ℝ,
      ⟦p⟧ = toGHSpace X →
        ⟦q⟧ = toGHSpace Y →
          hausdorffDist (range (optimalGHInjl X Y)) (range (optimalGHInjr X Y)) ≤
            hausdorffDist (p : Set ℓ_infty_ℝ) q := by
    intro p q hp hq
    by_cases h :
      hausdorffDist (p : Set ℓ_infty_ℝ) q < diam (univ : Set X) + 1 + diam (univ : Set Y)
    · exact A p q hp hq h
    · calc
        hausdorffDist (range (optimalGHInjl X Y)) (range (optimalGHInjr X Y)) ≤
            HD (candidatesBDist X Y) :=
          hausdorffDist_optimal_le_HD _ _ candidatesBDist_mem_candidatesB
        _ ≤ diam (univ : Set X) + 1 + diam (univ : Set Y) := HD_candidatesBDist_le
        _ ≤ hausdorffDist (p : Set ℓ_infty_ℝ) q := not_lt.1 h
  /-
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    inhabited_h✝ : Inhabited X
    inhabited_h : Inhabited Y
    A : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
    B : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
    ⊢ Eq (Metric.hausdorffDist (Set.range (GromovHausdorff.optimalGHInjl X Y)) (Se …
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      inhabited_h✝ : Inhabited X
      inhabited_h : Inhabited Y
      A : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
      B : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
      ⊢ LE.le (Metric.hausdorffDist (Set.range (GromovHausdorff.optimalGHInjl X Y))  …
    -/
  · apply le_csInf
      /-
        case refine_1.h₁
        X : Type u
        inst✝⁵ : MetricSpace X
        inst✝⁴ : CompactSpace X
        inst✝³ : Nonempty X
        Y : Type v
        inst✝² : MetricSpace Y
        inst✝¹ : CompactSpace Y
        inst✝ : Nonempty Y
        inhabited_h✝ : Inhabited X
        inhabited_h : Inhabited Y
        A : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
        B : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
        ⊢ (Set.image (fun p => Metric.hausdorffDist ↑p.1 ↑p.2) (SProd.sprod (setOf fun …
      -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    · refine (Set.Nonempty.prod ?_ ?_).image _ <;> exact ⟨_, rfl⟩
                                                   /-
                                                     🎉 no goals
                                                   -/
      /-
        case refine_1.h₂
        X : Type u
        inst✝⁵ : MetricSpace X
        inst✝⁴ : CompactSpace X
        inst✝³ : Nonempty X
        Y : Type v
        inst✝² : MetricSpace Y
        inst✝¹ : CompactSpace Y
        inst✝ : Nonempty Y
        inhabited_h✝ : Inhabited X
        inhabited_h : Inhabited Y
        A : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
        B : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
        ⊢ ∀ (b : Real), Membership.mem (Set.image (fun p => Metric.hausdorffDist ↑p.1  …
      -/
    · rintro b ⟨⟨p, q⟩, ⟨hp, hq⟩, rfl⟩
      /-
        case refine_1.h₂.intro.mk.intro.intro
        X : Type u
        inst✝⁵ : MetricSpace X
        inst✝⁴ : CompactSpace X
        inst✝³ : Nonempty X
        Y : Type v
        inst✝² : MetricSpace Y
        inst✝¹ : CompactSpace Y
        inst✝ : Nonempty Y
        inhabited_h✝ : Inhabited X
        inhabited_h : Inhabited Y
        A : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
        B : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
        p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp ( …
        hp : Membership.mem (setOf fun a => Eq (Quotient.mk GromovHausdorff.IsometryRe …
        hq : Membership.mem (setOf fun b => Eq (Quotient.mk GromovHausdorff.IsometryRe …
        ⊢ LE.le (Metric.hausdorffDist (Set.range (GromovHausdorff.optimalGHInjl X Y))  …
      -/
      exact B p q hp hq
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      inhabited_h✝ : Inhabited X
      inhabited_h : Inhabited Y
      A : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
      B : ∀ (p q : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.me …
      ⊢ LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range (GromovH …
    -/
  · exact ghDist_le_hausdorffDist (isometry_optimalGHInjl X Y) (isometry_optimalGHInjr X Y)
    /-
      🎉 no goals
    -/


/-- The Gromov-Hausdorff distance can also be realized by a coupling in `ℓ^∞(ℝ)`, by embedding
the optimal coupling through its Kuratowski embedding. -/
theorem ghDist_eq_hausdorffDist (X : Type u) [MetricSpace X] [CompactSpace X] [Nonempty X]
    (Y : Type v) [MetricSpace Y] [CompactSpace Y] [Nonempty Y] :
    ∃ Φ : X → ℓ_infty_ℝ,
      ∃ Ψ : Y → ℓ_infty_ℝ,
        Isometry Φ ∧ Isometry Ψ ∧ ghDist X Y = hausdorffDist (range Φ) (range Ψ) := by
  /-
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    ⊢ Exists fun Φ => Exists fun Ψ => And (Isometry Φ) (And (Isometry Ψ) (Eq (Grom …
  -/
  let F := kuratowskiEmbedding (OptimalGHCoupling X Y)
  /-
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    F : GromovHausdorff.OptimalGHCoupling X Y → Subtype fun x => Membership.mem (l …
    ⊢ Exists fun Φ => Exists fun Ψ => And (Isometry Φ) (And (Isometry Ψ) (Eq (Grom …
  -/
  let Φ := F ∘ optimalGHInjl X Y
  /-
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    F : GromovHausdorff.OptimalGHCoupling X Y → Subtype fun x => Membership.mem (l …
    Φ : X → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := Func …
    ⊢ Exists fun Φ => Exists fun Ψ => And (Isometry Φ) (And (Isometry Ψ) (Eq (Grom …
  -/
  let Ψ := F ∘ optimalGHInjr X Y
  /-
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    F : GromovHausdorff.OptimalGHCoupling X Y → Subtype fun x => Membership.mem (l …
    Φ : X → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := Func …
    Ψ : Y → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := Func …
    ⊢ Exists fun Φ => Exists fun Ψ => And (Isometry Φ) (And (Isometry Ψ) (Eq (Grom …
  -/
  refine ⟨Φ, Ψ, ?_, ?_, ?_⟩
    /-
      case refine_1
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      F : GromovHausdorff.OptimalGHCoupling X Y → Subtype fun x => Membership.mem (l …
      Φ : X → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := Func …
      Ψ : Y → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := Func …
      ⊢ Isometry Φ
    -/
  · exact (kuratowskiEmbedding.isometry _).comp (isometry_optimalGHInjl X Y)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      F : GromovHausdorff.OptimalGHCoupling X Y → Subtype fun x => Membership.mem (l …
      Φ : X → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := Func …
      Ψ : Y → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := Func …
      ⊢ Isometry Ψ
    -/
  · exact (kuratowskiEmbedding.isometry _).comp (isometry_optimalGHInjr X Y)
    /-
      🎉 no goals
    -/
  · rw [← image_univ, ← image_univ, image_comp F, image_univ, image_comp F (optimalGHInjr X Y),
      image_univ, ← hausdorffDist_optimal]
    /-
      case refine_3
      X : Type u
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      Y : Type v
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      F : GromovHausdorff.OptimalGHCoupling X Y → Subtype fun x => Membership.mem (l …
      Φ : X → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := Func …
      Ψ : Y → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := Func …
      ⊢ Eq (Metric.hausdorffDist (Set.range (GromovHausdorff.optimalGHInjl X Y)) (Se …
    -/
    exact (hausdorffDist_image (kuratowskiEmbedding.isometry _)).symm
    /-
      🎉 no goals
    -/


/-- The Gromov-Hausdorff distance defines a genuine distance on the Gromov-Hausdorff space. -/
instance : MetricSpace GHSpace where
  dist := dist
  dist_self x := by
    /-
      x : GromovHausdorff.GHSpace
      ⊢ Eq (Dist.dist x x) 0
    -/
    rcases exists_rep x with ⟨y, hy⟩
    /-
      case intro
      x : GromovHausdorff.GHSpace
      y : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
      hy : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid y) x
      ⊢ Eq (Dist.dist x x) 0
    -/
    refine le_antisymm ?_ ?_
      /-
        case intro.refine_1
        x : GromovHausdorff.GHSpace
        y : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
        hy : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid y) x
        ⊢ LE.le (Dist.dist x x) 0
      -/
    · apply csInf_le
        /-
          case intro.refine_1.h₁
          x : GromovHausdorff.GHSpace
          y : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
          hy : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid y) x
          ⊢ BddBelow (Set.image (fun p => Metric.hausdorffDist ↑p.1 ↑p.2) (SProd.sprod ( …
        -/
      · exact ⟨0, by rintro b ⟨⟨u, v⟩, -, rfl⟩; exact hausdorffDist_nonneg⟩
        /-
          🎉 no goals
        -/
        /-
          case intro.refine_1.h₂
          x : GromovHausdorff.GHSpace
          y : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
          hy : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid y) x
          ⊢ Membership.mem (Set.image (fun p => Metric.hausdorffDist ↑p.1 ↑p.2) (SProd.s …
        -/
      · simp only [mem_image, mem_prod, mem_setOf_eq, Prod.exists]
        /-
          case intro.refine_1.h₂
          x : GromovHausdorff.GHSpace
          y : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
          hy : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid y) x
          ⊢ Exists fun a => Exists fun b => And (And (Eq (Quotient.mk GromovHausdorff.Is …
        -/
        exists y, y
        /-
          case intro.refine_1.h₂
          x : GromovHausdorff.GHSpace
          y : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
          hy : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid y) x
          ⊢ And (And (Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid y) x) (Eq (Quot …
        -/
        simpa only [and_self_iff, hausdorffDist_self_zero, eq_self_iff_true, and_true]
        /-
          🎉 no goals
        -/
      /-
        case intro.refine_2
        x : GromovHausdorff.GHSpace
        y : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
        hy : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid y) x
        ⊢ LE.le 0 (Dist.dist x x)
      -/
    · apply le_csInf
        /-
          case intro.refine_2.h₁
          x : GromovHausdorff.GHSpace
          y : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
          hy : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid y) x
          ⊢ (Set.image (fun p => Metric.hausdorffDist ↑p.1 ↑p.2) (SProd.sprod (setOf fun …
        -/
      · exact Set.Nonempty.image _ <| Set.Nonempty.prod ⟨y, hy⟩ ⟨y, hy⟩
        /-
          🎉 no goals
        -/
        /-
          case intro.refine_2.h₂
          x : GromovHausdorff.GHSpace
          y : TopologicalSpace.NonemptyCompacts (Subtype fun x => Membership.mem (lp (fu …
          hy : Eq (Quotient.mk GromovHausdorff.IsometryRel.setoid y) x
          ⊢ ∀ (b : Real), Membership.mem (Set.image (fun p => Metric.hausdorffDist ↑p.1  …
        -/
      · rintro b ⟨⟨u, v⟩, -, rfl⟩; exact hausdorffDist_nonneg
                                   /-
                                     🎉 no goals
                                   -/
  dist_comm x y := by
    have A :
      (fun p : NonemptyCompacts ℓ_infty_ℝ × NonemptyCompacts ℓ_infty_ℝ =>
            hausdorffDist (p.1 : Set ℓ_infty_ℝ) p.2) ''
          { a | ⟦a⟧ = x } ×ˢ { b | ⟦b⟧ = y } =
        (fun p : NonemptyCompacts ℓ_infty_ℝ × NonemptyCompacts ℓ_infty_ℝ =>
              hausdorffDist (p.1 : Set ℓ_infty_ℝ) p.2) ∘
            Prod.swap ''
          { a | ⟦a⟧ = x } ×ˢ { b | ⟦b⟧ = y } := by
      funext
      simp only [comp_apply, Prod.fst_swap, Prod.snd_swap]
      congr
      -- The next line had `singlePass := true` before https://github.com/leanprover-community/mathlib4/pull/9928,
      -- then was changed to be `simp only [hausdorffDist_comm]`,
      -- then `singlePass := true` was readded in https://github.com/leanprover-community/mathlib4/pull/8386 because of timeouts.
      -- TODO: figure out what causes the slowdown and make it a `simp only` again?
      simp (config := { singlePass := true }) only [hausdorffDist_comm]
    /-
      x y : GromovHausdorff.GHSpace
      A : Eq (Set.image (fun p => Metric.hausdorffDist ↑p.1 ↑p.2) (SProd.sprod (setO …
      ⊢ Eq (Dist.dist x y) (Dist.dist y x)
    -/
    simp only [dist, A, image_comp, image_swap_prod]
    /-
      🎉 no goals
    -/
  eq_of_dist_eq_zero {x} {y} hxy := by
    /- To show that two spaces at zero distance are isometric,
       we argue that the distance is realized by some coupling.
        In this coupling, the two spaces are at zero Hausdorff distance,
        i.e., they coincide. Therefore, the original spaces are isometric. -/
    /-
      x y : GromovHausdorff.GHSpace
      hxy : Eq (Dist.dist x y) 0
      ⊢ Eq x y
    -/
    rcases ghDist_eq_hausdorffDist x.Rep y.Rep with ⟨Φ, Ψ, Φisom, Ψisom, DΦΨ⟩
    /-
      case intro.intro.intro.intro
      x y : GromovHausdorff.GHSpace
      hxy : Eq (Dist.dist x y) 0
      Φ : x.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Ψ : y.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Φisom : Isometry Φ
      Ψisom : Isometry Ψ
      DΦΨ : Eq (GromovHausdorff.ghDist x.Rep y.Rep) (Metric.hausdorffDist (Set.range …
      ⊢ Eq x y
    -/
    rw [← dist_ghDist] at DΦΨ
    /-
      case intro.intro.intro.intro
      x y : GromovHausdorff.GHSpace
      hxy : Eq (Dist.dist x y) 0
      Φ : x.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Ψ : y.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Φisom : Isometry Φ
      Ψisom : Isometry Ψ
      DΦΨ : Eq (Dist.dist x y) (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ))
      ⊢ Eq x y
    -/
    simp_rw [hxy] at DΦΨ -- Porting note: I have no idea why this needed `simp_rw` versus `rw`
    have : range Φ = range Ψ := by
      have hΦ : IsCompact (range Φ) := isCompact_range Φisom.continuous
      have hΨ : IsCompact (range Ψ) := isCompact_range Ψisom.continuous
      apply (IsClosed.hausdorffDist_zero_iff_eq _ _ _).1 DΦΨ.symm
      · exact hΦ.isClosed
      · exact hΨ.isClosed
      · exact hausdorffEdist_ne_top_of_nonempty_of_bounded (range_nonempty _) (range_nonempty _)
          hΦ.isBounded hΨ.isBounded
    /-
      case intro.intro.intro.intro
      x y : GromovHausdorff.GHSpace
      hxy : Eq (Dist.dist x y) 0
      Φ : x.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Ψ : y.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Φisom : Isometry Φ
      Ψisom : Isometry Ψ
      DΦΨ : Eq 0 (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ))
      this : Eq (Set.range Φ) (Set.range Ψ)
      ⊢ Eq x y
    -/
    have T : (range Ψ ≃ᵢ y.Rep) = (range Φ ≃ᵢ y.Rep) := by rw [this]
    /-
      case intro.intro.intro.intro
      x y : GromovHausdorff.GHSpace
      hxy : Eq (Dist.dist x y) 0
      Φ : x.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Ψ : y.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Φisom : Isometry Φ
      Ψisom : Isometry Ψ
      DΦΨ : Eq 0 (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ))
      this : Eq (Set.range Φ) (Set.range Ψ)
      T : Eq (IsometryEquiv (↑(Set.range Ψ)) y.Rep) (IsometryEquiv (↑(Set.range Φ))  …
      ⊢ Eq x y
    -/
    have eΨ := cast T Ψisom.isometryEquivOnRange.symm
    /-
      case intro.intro.intro.intro
      x y : GromovHausdorff.GHSpace
      hxy : Eq (Dist.dist x y) 0
      Φ : x.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Ψ : y.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Φisom : Isometry Φ
      Ψisom : Isometry Ψ
      DΦΨ : Eq 0 (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ))
      this : Eq (Set.range Φ) (Set.range Ψ)
      T : Eq (IsometryEquiv (↑(Set.range Ψ)) y.Rep) (IsometryEquiv (↑(Set.range Φ))  …
      eΨ : IsometryEquiv (↑(Set.range Φ)) y.Rep
      ⊢ Eq x y
    -/
    have e := Φisom.isometryEquivOnRange.trans eΨ
    /-
      case intro.intro.intro.intro
      x y : GromovHausdorff.GHSpace
      hxy : Eq (Dist.dist x y) 0
      Φ : x.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Ψ : y.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Φisom : Isometry Φ
      Ψisom : Isometry Ψ
      DΦΨ : Eq 0 (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ))
      this : Eq (Set.range Φ) (Set.range Ψ)
      T : Eq (IsometryEquiv (↑(Set.range Ψ)) y.Rep) (IsometryEquiv (↑(Set.range Φ))  …
      eΨ : IsometryEquiv (↑(Set.range Φ)) y.Rep
      e : IsometryEquiv x.Rep y.Rep
      ⊢ Eq x y
    -/
    rw [← x.toGHSpace_rep, ← y.toGHSpace_rep, toGHSpace_eq_toGHSpace_iff_isometryEquiv]
    /-
      case intro.intro.intro.intro
      x y : GromovHausdorff.GHSpace
      hxy : Eq (Dist.dist x y) 0
      Φ : x.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Ψ : y.Rep → Subtype fun x => Membership.mem (lp (fun n => Real) Top.top) x
      Φisom : Isometry Φ
      Ψisom : Isometry Ψ
      DΦΨ : Eq 0 (Metric.hausdorffDist (Set.range Φ) (Set.range Ψ))
      this : Eq (Set.range Φ) (Set.range Ψ)
      T : Eq (IsometryEquiv (↑(Set.range Ψ)) y.Rep) (IsometryEquiv (↑(Set.range Φ))  …
      eΨ : IsometryEquiv (↑(Set.range Φ)) y.Rep
      e : IsometryEquiv x.Rep y.Rep
      ⊢ Nonempty (IsometryEquiv x.Rep y.Rep)
    -/
    exact ⟨e⟩
    /-
      x y z : GromovHausdorff.GHSpace
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
    /-
      🎉 no goals
    -/
    /-
      x y z : GromovHausdorff.GHSpace
      X : Type := x.Rep
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
  dist_triangle x y z := by
    /-
      x y z : GromovHausdorff.GHSpace
      X : Type := x.Rep
      Y : Type := y.Rep
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
    /- To show the triangular inequality between `X`, `Y` and `Z`,
    /-
      x y z : GromovHausdorff.GHSpace
      X : Type := x.Rep
      Y : Type := y.Rep
      Z : Type := z.Rep
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
        realize an optimal coupling between `X` and `Y` in a space `γ1`,
    /-
      x y z : GromovHausdorff.GHSpace
      X : Type := x.Rep
      Y : Type := y.Rep
      Z : Type := z.Rep
      γ1 : Type := GromovHausdorff.OptimalGHCoupling X Y
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
        and an optimal coupling between `Y` and `Z` in a space `γ2`.
    /-
      x y z : GromovHausdorff.GHSpace
      X : Type := x.Rep
      Y : Type := y.Rep
      Z : Type := z.Rep
      γ1 : Type := GromovHausdorff.OptimalGHCoupling X Y
      γ2 : Type := GromovHausdorff.OptimalGHCoupling Y Z
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
        Then, glue these metric spaces along `Y`. We get a new space `γ`
    /-
      x y z : GromovHausdorff.GHSpace
      X : Type := x.Rep
      Y : Type := y.Rep
      Z : Type := z.Rep
      γ1 : Type := GromovHausdorff.OptimalGHCoupling X Y
      γ2 : Type := GromovHausdorff.OptimalGHCoupling Y Z
      Φ : Y → γ1 := GromovHausdorff.optimalGHInjr X Y
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
        in which `X` and `Y` are optimally coupled, as well as `Y` and `Z`.
    /-
      x y z : GromovHausdorff.GHSpace
      X : Type := x.Rep
      Y : Type := y.Rep
      Z : Type := z.Rep
      γ1 : Type := GromovHausdorff.OptimalGHCoupling X Y
      γ2 : Type := GromovHausdorff.OptimalGHCoupling Y Z
      Φ : Y → γ1 := GromovHausdorff.optimalGHInjr X Y
      hΦ : Isometry Φ
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
        Apply the triangle inequality for the Hausdorff distance in `γ`
    /-
      x y z : GromovHausdorff.GHSpace
      X : Type := x.Rep
      Y : Type := y.Rep
      Z : Type := z.Rep
      γ1 : Type := GromovHausdorff.OptimalGHCoupling X Y
      γ2 : Type := GromovHausdorff.OptimalGHCoupling Y Z
      Φ : Y → γ1 := GromovHausdorff.optimalGHInjr X Y
      hΦ : Isometry Φ
      Ψ : Y → γ2 := GromovHausdorff.optimalGHInjl Y Z
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
        to conclude. -/
    let X := x.Rep
    let Y := y.Rep
    let Z := z.Rep
    let γ1 := OptimalGHCoupling X Y
    let γ2 := OptimalGHCoupling Y Z
    let Φ : Y → γ1 := optimalGHInjr X Y
    have hΦ : Isometry Φ := isometry_optimalGHInjr X Y
    let Ψ : Y → γ2 := optimalGHInjl Y Z
    have hΨ : Isometry Ψ := isometry_optimalGHInjl Y Z
    have Comm : toGlueL hΦ hΨ ∘ optimalGHInjr X Y = toGlueR hΦ hΨ ∘ optimalGHInjl Y Z :=
      toGlue_commute hΦ hΨ
    calc
      dist x z = dist (toGHSpace X) (toGHSpace Z) := by
        rw [x.toGHSpace_rep, z.toGHSpace_rep]
      _ ≤ hausdorffDist (range (toGlueL hΦ hΨ ∘ optimalGHInjl X Y))
            (range (toGlueR hΦ hΨ ∘ optimalGHInjr Y Z)) :=
        (ghDist_le_hausdorffDist ((toGlueL_isometry hΦ hΨ).comp (isometry_optimalGHInjl X Y))
          ((toGlueR_isometry hΦ hΨ).comp (isometry_optimalGHInjr Y Z)))
      _ ≤ hausdorffDist (range (toGlueL hΦ hΨ ∘ optimalGHInjl X Y))
              (range (toGlueL hΦ hΨ ∘ optimalGHInjr X Y)) +
            hausdorffDist (range (toGlueL hΦ hΨ ∘ optimalGHInjr X Y))
              (range (toGlueR hΦ hΨ ∘ optimalGHInjr Y Z)) := by
        refine hausdorffDist_triangle <| hausdorffEdist_ne_top_of_nonempty_of_bounded
          (range_nonempty _) (range_nonempty _) ?_ ?_
        · exact (isCompact_range (Isometry.continuous
            ((toGlueL_isometry hΦ hΨ).comp (isometry_optimalGHInjl X Y)))).isBounded
        · exact (isCompact_range (Isometry.continuous
            ((toGlueL_isometry hΦ hΨ).comp (isometry_optimalGHInjr X Y)))).isBounded
      _ = hausdorffDist (toGlueL hΦ hΨ '' range (optimalGHInjl X Y))
              (toGlueL hΦ hΨ '' range (optimalGHInjr X Y)) +
            hausdorffDist (toGlueR hΦ hΨ '' range (optimalGHInjl Y Z))
              (toGlueR hΦ hΨ '' range (optimalGHInjr Y Z)) := by
        simp only [← range_comp, Comm, eq_self_iff_true, add_right_inj]
      _ = hausdorffDist (range (optimalGHInjl X Y)) (range (optimalGHInjr X Y)) +
            hausdorffDist (range (optimalGHInjl Y Z)) (range (optimalGHInjr Y Z)) := by
        rw [hausdorffDist_image (toGlueL_isometry hΦ hΨ),
          hausdorffDist_image (toGlueR_isometry hΦ hΨ)]
      _ = dist (toGHSpace X) (toGHSpace Y) + dist (toGHSpace Y) (toGHSpace Z) := by
        rw [hausdorffDist_optimal, hausdorffDist_optimal, ghDist, ghDist]
      _ = dist x y + dist y z := by rw [x.toGHSpace_rep, y.toGHSpace_rep, z.toGHSpace_rep]



/-- In particular, nonempty compacts of a metric space map to `GHSpace`.
    We register this in the `TopologicalSpace` namespace to take advantage
    of the notation `p.toGHSpace`. -/
def TopologicalSpace.NonemptyCompacts.toGHSpace {X : Type u} [MetricSpace X]
    (p : NonemptyCompacts X) : GromovHausdorff.GHSpace :=
  GromovHausdorff.toGHSpace p


theorem ghDist_le_nonemptyCompacts_dist (p q : NonemptyCompacts X) :
    dist p.toGHSpace q.toGHSpace ≤ dist p q := by
  /-
    X : Type u
    inst✝ : MetricSpace X
    p q : TopologicalSpace.NonemptyCompacts X
    ⊢ LE.le (Dist.dist p.toGHSpace q.toGHSpace) (Dist.dist p q)
  -/
  have ha : Isometry ((↑) : p → X) := isometry_subtype_coe
  /-
    X : Type u
    inst✝ : MetricSpace X
    p q : TopologicalSpace.NonemptyCompacts X
    ha : Isometry Subtype.val
    ⊢ LE.le (Dist.dist p.toGHSpace q.toGHSpace) (Dist.dist p q)
  -/
  have hb : Isometry ((↑) : q → X) := isometry_subtype_coe
  /-
    X : Type u
    inst✝ : MetricSpace X
    p q : TopologicalSpace.NonemptyCompacts X
    ha : Isometry Subtype.val
    hb : Isometry Subtype.val
    ⊢ LE.le (Dist.dist p.toGHSpace q.toGHSpace) (Dist.dist p q)
  -/
  have A : dist p q = hausdorffDist (p : Set X) q := rfl
  /-
    X : Type u
    inst✝ : MetricSpace X
    p q : TopologicalSpace.NonemptyCompacts X
    ha : Isometry Subtype.val
    hb : Isometry Subtype.val
    A : Eq (Dist.dist p q) (Metric.hausdorffDist ↑p ↑q)
    ⊢ LE.le (Dist.dist p.toGHSpace q.toGHSpace) (Dist.dist p q)
  -/
  have I : ↑p = range ((↑) : p → X) := Subtype.range_coe_subtype.symm
  /-
    X : Type u
    inst✝ : MetricSpace X
    p q : TopologicalSpace.NonemptyCompacts X
    ha : Isometry Subtype.val
    hb : Isometry Subtype.val
    A : Eq (Dist.dist p q) (Metric.hausdorffDist ↑p ↑q)
    I : Eq (↑p) (Set.range Subtype.val)
    ⊢ LE.le (Dist.dist p.toGHSpace q.toGHSpace) (Dist.dist p q)
  -/
  have J : ↑q = range ((↑) : q → X) := Subtype.range_coe_subtype.symm
  /-
    X : Type u
    inst✝ : MetricSpace X
    p q : TopologicalSpace.NonemptyCompacts X
    ha : Isometry Subtype.val
    hb : Isometry Subtype.val
    A : Eq (Dist.dist p q) (Metric.hausdorffDist ↑p ↑q)
    I : Eq (↑p) (Set.range Subtype.val)
    J : Eq (↑q) (Set.range Subtype.val)
    ⊢ LE.le (Dist.dist p.toGHSpace q.toGHSpace) (Dist.dist p q)
  -/
  rw [A, I, J]
  /-
    X : Type u
    inst✝ : MetricSpace X
    p q : TopologicalSpace.NonemptyCompacts X
    ha : Isometry Subtype.val
    hb : Isometry Subtype.val
    A : Eq (Dist.dist p q) (Metric.hausdorffDist ↑p ↑q)
    I : Eq (↑p) (Set.range Subtype.val)
    J : Eq (↑q) (Set.range Subtype.val)
    ⊢ LE.le (Dist.dist p.toGHSpace q.toGHSpace) (Metric.hausdorffDist (Set.range S …
  -/
  exact ghDist_le_hausdorffDist ha hb
  /-
    🎉 no goals
  -/


theorem toGHSpace_lipschitz :
    LipschitzWith 1 (NonemptyCompacts.toGHSpace : NonemptyCompacts X → GHSpace) :=
  LipschitzWith.mk_one ghDist_le_nonemptyCompacts_dist


theorem toGHSpace_continuous :
    Continuous (NonemptyCompacts.toGHSpace : NonemptyCompacts X → GHSpace) :=
  toGHSpace_lipschitz.continuous


/-- If there are subsets which are `ε₁`-dense and `ε₃`-dense in two spaces, and
isometric up to `ε₂`, then the Gromov-Hausdorff distance between the spaces is bounded by
`ε₁ + ε₂/2 + ε₃`. -/
theorem ghDist_le_of_approx_subsets {s : Set X} (Φ : s → Y) {ε₁ ε₂ ε₃ : ℝ}
    (hs : ∀ x : X, ∃ y ∈ s, dist x y ≤ ε₁) (hs' : ∀ x : Y, ∃ y : s, dist x (Φ y) ≤ ε₃)
    (H : ∀ x y : s, |dist x y - dist (Φ x) (Φ y)| ≤ ε₂) : ghDist X Y ≤ ε₁ + ε₂ / 2 + ε₃ := by
  /-
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv.hDiv ε₂ 2) …
  -/
  refine le_of_forall_pos_le_add fun δ δ0 => ?_
  /-
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    δ : Real
    δ0 : LT.lt 0 δ
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv …
  -/
  rcases exists_mem_of_nonempty X with ⟨xX, _⟩
  /-
    case intro
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    δ : Real
    δ0 : LT.lt 0 δ
    xX : X
    h✝ : Membership.mem Set.univ xX
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv …
  -/
  rcases hs xX with ⟨xs, hxs, Dxs⟩
  /-
    case intro.intro.intro
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    δ : Real
    δ0 : LT.lt 0 δ
    xX : X
    h✝ : Membership.mem Set.univ xX
    xs : X
    hxs : Membership.mem s xs
    Dxs : LE.le (Dist.dist xX xs) ε₁
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv …
  -/
  have sne : s.Nonempty := ⟨xs, hxs⟩
  /-
    case intro.intro.intro
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    δ : Real
    δ0 : LT.lt 0 δ
    xX : X
    h✝ : Membership.mem Set.univ xX
    xs : X
    hxs : Membership.mem s xs
    Dxs : LE.le (Dist.dist xX xs) ε₁
    sne : s.Nonempty
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv …
  -/
  letI : Nonempty s := sne.to_subtype
  /-
    case intro.intro.intro
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    δ : Real
    δ0 : LT.lt 0 δ
    xX : X
    h✝ : Membership.mem Set.univ xX
    xs : X
    hxs : Membership.mem s xs
    Dxs : LE.le (Dist.dist xX xs) ε₁
    sne : s.Nonempty
    this : Nonempty ↑s := Set.Nonempty.to_subtype sne
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv …
  -/
  have : 0 ≤ ε₂ := le_trans (abs_nonneg _) (H ⟨xs, hxs⟩ ⟨xs, hxs⟩)
  have : ∀ p q : s, |dist p q - dist (Φ p) (Φ q)| ≤ 2 * (ε₂ / 2 + δ) := fun p q =>
    calc
      |dist p q - dist (Φ p) (Φ q)| ≤ ε₂ := H p q
      _ ≤ 2 * (ε₂ / 2 + δ) := by linarith
  -- glue `X` and `Y` along the almost matching subsets
  letI : MetricSpace (X ⊕ Y) :=
    glueMetricApprox (fun x : s => (x : X)) (fun x => Φ x) (ε₂ / 2 + δ) (by linarith) this
  /-
    case intro.intro.intro
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    δ : Real
    δ0 : LT.lt 0 δ
    xX : X
    h✝ : Membership.mem Set.univ xX
    xs : X
    hxs : Membership.mem s xs
    Dxs : LE.le (Dist.dist xX xs) ε₁
    sne : s.Nonempty
    this✝² : Nonempty ↑s := Set.Nonempty.to_subtype sne
    this✝¹ : LE.le 0 ε₂
    this✝ : ∀ (p q : ↑s), LE.le (abs (HSub.hSub (Dist.dist p q) (Dist.dist (Φ p) ( …
    this : MetricSpace (Sum X Y) := Metric.glueMetricApprox (fun x => ↑x) (fun x = …
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv …
  -/
  let Fl := @Sum.inl X Y
  /-
    case intro.intro.intro
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    δ : Real
    δ0 : LT.lt 0 δ
    xX : X
    h✝ : Membership.mem Set.univ xX
    xs : X
    hxs : Membership.mem s xs
    Dxs : LE.le (Dist.dist xX xs) ε₁
    sne : s.Nonempty
    this✝² : Nonempty ↑s := Set.Nonempty.to_subtype sne
    this✝¹ : LE.le 0 ε₂
    this✝ : ∀ (p q : ↑s), LE.le (abs (HSub.hSub (Dist.dist p q) (Dist.dist (Φ p) ( …
    this : MetricSpace (Sum X Y) := Metric.glueMetricApprox (fun x => ↑x) (fun x = …
    Fl : X → Sum X Y := Sum.inl
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv …
  -/
  let Fr := @Sum.inr X Y
  /-
    case intro.intro.intro
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    δ : Real
    δ0 : LT.lt 0 δ
    xX : X
    h✝ : Membership.mem Set.univ xX
    xs : X
    hxs : Membership.mem s xs
    Dxs : LE.le (Dist.dist xX xs) ε₁
    sne : s.Nonempty
    this✝² : Nonempty ↑s := Set.Nonempty.to_subtype sne
    this✝¹ : LE.le 0 ε₂
    this✝ : ∀ (p q : ↑s), LE.le (abs (HSub.hSub (Dist.dist p q) (Dist.dist (Φ p) ( …
    this : MetricSpace (Sum X Y) := Metric.glueMetricApprox (fun x => ↑x) (fun x = …
    Fl : X → Sum X Y := Sum.inl
    Fr : Y → Sum X Y := Sum.inr
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv …
  -/
  have Il : Isometry Fl := Isometry.of_dist_eq fun x y => rfl
  /-
    case intro.intro.intro
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    δ : Real
    δ0 : LT.lt 0 δ
    xX : X
    h✝ : Membership.mem Set.univ xX
    xs : X
    hxs : Membership.mem s xs
    Dxs : LE.le (Dist.dist xX xs) ε₁
    sne : s.Nonempty
    this✝² : Nonempty ↑s := Set.Nonempty.to_subtype sne
    this✝¹ : LE.le 0 ε₂
    this✝ : ∀ (p q : ↑s), LE.le (abs (HSub.hSub (Dist.dist p q) (Dist.dist (Φ p) ( …
    this : MetricSpace (Sum X Y) := Metric.glueMetricApprox (fun x => ↑x) (fun x = …
    Fl : X → Sum X Y := Sum.inl
    Fr : Y → Sum X Y := Sum.inr
    Il : Isometry Fl
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv …
  -/
  have Ir : Isometry Fr := Isometry.of_dist_eq fun x y => rfl
  /- The proof goes as follows : the `GH_dist` is bounded by the Hausdorff distance of the images
    in the coupling, which is bounded (using the triangular inequality) by the sum of the Hausdorff
    distances of `X` and `s` (in the coupling or, equivalently in the original space), of `s` and
    `Φ s`, and of `Φ s` and `Y` (in the coupling or, equivalently, in the original space).
    The first term is bounded by `ε₁`, by `ε₁`-density. The third one is bounded by `ε₃`.
    And the middle one is bounded by `ε₂/2` as in the coupling the points `x` and `Φ x` are
    at distance `ε₂/2` by construction of the coupling (in fact `ε₂/2 + δ` where `δ` is an
    arbitrarily small positive constant where positivity is used to ensure that the coupling
    is really a metric space and not a premetric space on `X ⊕ Y`). -/
  /-
    case intro.intro.intro
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    δ : Real
    δ0 : LT.lt 0 δ
    xX : X
    h✝ : Membership.mem Set.univ xX
    xs : X
    hxs : Membership.mem s xs
    Dxs : LE.le (Dist.dist xX xs) ε₁
    sne : s.Nonempty
    this✝² : Nonempty ↑s := Set.Nonempty.to_subtype sne
    this✝¹ : LE.le 0 ε₂
    this✝ : ∀ (p q : ↑s), LE.le (abs (HSub.hSub (Dist.dist p q) (Dist.dist (Φ p) ( …
    this : MetricSpace (Sum X Y) := Metric.glueMetricApprox (fun x => ↑x) (fun x = …
    Fl : X → Sum X Y := Sum.inl
    Fr : Y → Sum X Y := Sum.inr
    Il : Isometry Fl
    Ir : Isometry Fr
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv …
  -/
  have : ghDist X Y ≤ hausdorffDist (range Fl) (range Fr) := ghDist_le_hausdorffDist Il Ir
  have :
    hausdorffDist (range Fl) (range Fr) ≤
      hausdorffDist (range Fl) (Fl '' s) + hausdorffDist (Fl '' s) (range Fr) :=
    have B : IsBounded (range Fl) := (isCompact_range Il.continuous).isBounded
    hausdorffDist_triangle
      (hausdorffEdist_ne_top_of_nonempty_of_bounded (range_nonempty _) (sne.image _) B
        (B.subset (image_subset_range _ _)))
  have :
    hausdorffDist (Fl '' s) (range Fr) ≤
      hausdorffDist (Fl '' s) (Fr '' range Φ) + hausdorffDist (Fr '' range Φ) (range Fr) :=
    have B : IsBounded (range Fr) := (isCompact_range Ir.continuous).isBounded
    hausdorffDist_triangle'
      (hausdorffEdist_ne_top_of_nonempty_of_bounded ((range_nonempty _).image _) (range_nonempty _)
        (B.subset (image_subset_range _ _)) B)
  have : hausdorffDist (range Fl) (Fl '' s) ≤ ε₁ := by
    rw [← image_univ, hausdorffDist_image Il]
    have : 0 ≤ ε₁ := le_trans dist_nonneg Dxs
    refine hausdorffDist_le_of_mem_dist this (fun x _ => hs x) fun x _ =>
      ⟨x, mem_univ _, by simpa only [dist_self]⟩
  have : hausdorffDist (Fl '' s) (Fr '' range Φ) ≤ ε₂ / 2 + δ := by
    refine hausdorffDist_le_of_mem_dist (by linarith) ?_ ?_
    · intro x' hx'
      rcases (Set.mem_image _ _ _).1 hx' with ⟨x, ⟨x_in_s, xx'⟩⟩
      rw [← xx']
      use Fr (Φ ⟨x, x_in_s⟩), mem_image_of_mem Fr (mem_range_self _)
      exact le_of_eq (glueDist_glued_points (fun x : s => (x : X)) Φ (ε₂ / 2 + δ) ⟨x, x_in_s⟩)
    · intro x' hx'
      rcases (Set.mem_image _ _ _).1 hx' with ⟨y, ⟨y_in_s', yx'⟩⟩
      rcases mem_range.1 y_in_s' with ⟨x, xy⟩
      use Fl x, mem_image_of_mem _ x.2
      rw [← yx', ← xy, dist_comm]
      exact le_of_eq (glueDist_glued_points (Z := s) (@Subtype.val X s) Φ (ε₂ / 2 + δ) x)
  have : hausdorffDist (Fr '' range Φ) (range Fr) ≤ ε₃ := by
    rw [← @image_univ _ _ Fr, hausdorffDist_image Ir]
    rcases exists_mem_of_nonempty Y with ⟨xY, _⟩
    rcases hs' xY with ⟨xs', Dxs'⟩
    have : 0 ≤ ε₃ := le_trans dist_nonneg Dxs'
    refine hausdorffDist_le_of_mem_dist this
      (fun x _ => ⟨x, mem_univ _, by simpa only [dist_self]⟩)
      fun x _ => ?_
    rcases hs' x with ⟨y, Dy⟩
    exact ⟨Φ y, mem_range_self _, Dy⟩
  /-
    case intro.intro.intro
    X : Type u
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    Y : Type v
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    s : Set X
    Φ : ↑s → Y
    ε₁ ε₂ ε₃ : Real
    hs : ∀ (x : X), Exists fun y => And (Membership.mem s y) (LE.le (Dist.dist x y …
    hs' : ∀ (x : Y), Exists fun y => LE.le (Dist.dist x (Φ y)) ε₃
    H : ∀ (x y : ↑s), LE.le (abs (HSub.hSub (Dist.dist x y) (Dist.dist (Φ x) (Φ y) …
    δ : Real
    δ0 : LT.lt 0 δ
    xX : X
    h✝ : Membership.mem Set.univ xX
    xs : X
    hxs : Membership.mem s xs
    Dxs : LE.le (Dist.dist xX xs) ε₁
    sne : s.Nonempty
    this✝⁸ : Nonempty ↑s := Set.Nonempty.to_subtype sne
    this✝⁷ : LE.le 0 ε₂
    this✝⁶ : ∀ (p q : ↑s), LE.le (abs (HSub.hSub (Dist.dist p q) (Dist.dist (Φ p)  …
    this✝⁵ : MetricSpace (Sum X Y) := Metric.glueMetricApprox (fun x => ↑x) (fun x …
    Fl : X → Sum X Y := Sum.inl
    Fr : Y → Sum X Y := Sum.inr
    Il : Isometry Fl
    Ir : Isometry Fr
    this✝⁴ : LE.le (GromovHausdorff.ghDist X Y) (Metric.hausdorffDist (Set.range F …
    this✝³ : LE.le (Metric.hausdorffDist (Set.range Fl) (Set.range Fr)) (HAdd.hAdd …
    this✝² : LE.le (Metric.hausdorffDist (Set.image Fl s) (Set.range Fr)) (HAdd.hA …
    this✝¹ : LE.le (Metric.hausdorffDist (Set.range Fl) (Set.image Fl s)) ε₁
    this✝ : LE.le (Metric.hausdorffDist (Set.image Fl s) (Set.image Fr (Set.range  …
    this : LE.le (Metric.hausdorffDist (Set.image Fr (Set.range Φ)) (Set.range Fr) …
    ⊢ LE.le (GromovHausdorff.ghDist X Y) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ε₁ (HDiv …
  -/
  linarith
  /-
    🎉 no goals
  -/


/-- The Gromov-Hausdorff space is second countable. -/
instance : SecondCountableTopology GHSpace := by
  /-
    ⊢ SecondCountableTopology GromovHausdorff.GHSpace
  -/
  refine secondCountable_of_countable_discretization fun δ δpos => ?_
  /-
    δ : Real
    δpos : GT.gt δ 0
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : GromovHausdorff.GHS …
  -/
  let ε := 2 / 5 * δ
  /-
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (2 / 5) δ
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : GromovHausdorff.GHS …
  -/
  have εpos : 0 < ε := mul_pos (by norm_num) δpos
  have : ∀ p : GHSpace, ∃ s : Set p.Rep, s.Finite ∧ univ ⊆ ⋃ x ∈ s, ball x ε := fun p => by
    simpa only [subset_univ, true_and] using
      finite_cover_balls_of_compact (α := p.Rep) isCompact_univ εpos
  -- for each `p`, `s p` is a finite `ε`-dense subset of `p` (or rather the metric space
  -- `p.rep` representing `p`)
  /-
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (2 / 5) δ
    εpos : LT.lt 0 ε
    this : ∀ (p : GromovHausdorff.GHSpace), Exists fun s => And s.Finite (HasSubse …
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : GromovHausdorff.GHS …
  -/
  choose s hs using this
  have : ∀ p : GHSpace, ∀ t : Set p.Rep, t.Finite → ∃ n : ℕ, ∃ _ : Equiv t (Fin n), True := by
    intro p t ht
    letI : Fintype t := Finite.fintype ht
    exact ⟨Fintype.card t, Fintype.equivFin t, trivial⟩
  /-
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (2 / 5) δ
    εpos : LT.lt 0 ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    hs : ∀ (p : GromovHausdorff.GHSpace), And (s p).Finite (HasSubset.Subset Set.u …
    this : ∀ (p : GromovHausdorff.GHSpace) (t : Set p.Rep), t.Finite → Exists fun  …
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : GromovHausdorff.GHS …
  -/
  choose N e _ using this
  -- cardinality of the nice finite subset `s p` of `p.rep`, called `N p`
  /-
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (2 / 5) δ
    εpos : LT.lt 0 ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    hs : ∀ (p : GromovHausdorff.GHSpace), And (s p).Finite (HasSubset.Subset Set.u …
    N : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → t.Finite → Nat
    e : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → (a : t.Finite) → Equiv ( …
    x✝ : ∀ (p : GromovHausdorff.GHSpace) (t : Set p.Rep), t.Finite → True
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : GromovHausdorff.GHS …
  -/
  let N := fun p : GHSpace => N p (s p) (hs p).1
  -- equiv from `s p`, a nice finite subset of `p.rep`, to `Fin (N p)`, called `E p`
  /-
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (2 / 5) δ
    εpos : LT.lt 0 ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    hs : ∀ (p : GromovHausdorff.GHSpace), And (s p).Finite (HasSubset.Subset Set.u …
    N✝ : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → t.Finite → Nat
    e : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → (a : t.Finite) → Equiv ( …
    x✝ : ∀ (p : GromovHausdorff.GHSpace) (t : Set p.Rep), t.Finite → True
    N : GromovHausdorff.GHSpace → Nat := fun p => N✝ p (s p) ⋯
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : GromovHausdorff.GHS …
  -/
  let E := fun p : GHSpace => e p (s p) (hs p).1
  -- A function `F` associating to `p : GHSpace` the data of all distances between points
  -- in the `ε`-dense set `s p`.
  let F : GHSpace → Σ n : ℕ, Fin n → Fin n → ℤ := fun p =>
    ⟨N p, fun a b => ⌊ε⁻¹ * dist ((E p).symm a) ((E p).symm b)⌋⟩
  /-
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (2 / 5) δ
    εpos : LT.lt 0 ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    hs : ∀ (p : GromovHausdorff.GHSpace), And (s p).Finite (HasSubset.Subset Set.u …
    N✝ : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → t.Finite → Nat
    e : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → (a : t.Finite) → Equiv ( …
    x✝ : ∀ (p : GromovHausdorff.GHSpace) (t : Set p.Rep), t.Finite → True
    N : GromovHausdorff.GHSpace → Nat := fun p => N✝ p (s p) ⋯
    E : (p : GromovHausdorff.GHSpace) → Equiv (↑(s p)) (Fin (N✝ p (s p) ⋯)) := fun …
    F : GromovHausdorff.GHSpace → Sigma fun n => Fin n → Fin n → Int := fun p => ⟨ …
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : GromovHausdorff.GHS …
  -/
  refine ⟨Σ n, Fin n → Fin n → ℤ, by infer_instance, F, fun p q hpq => ?_⟩
  /- As the target space of F is countable, it suffices to show that two points
    `p` and `q` with `F p = F q` are at distance `≤ δ`.
    For this, we construct a map `Φ` from `s p ⊆ p.rep` (representing `p`)
    to `q.rep` (representing `q`) which is almost an isometry on `s p`, and
    with image `s q`. For this, we compose the identification of `s p` with `Fin (N p)`
    and the inverse of the identification of `s q` with `Fin (N q)`. Together with
    the fact that `N p = N q`, this constructs `Ψ` between `s p` and `s q`, and then
    composing with the canonical inclusion we get `Φ`. -/
  /-
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (2 / 5) δ
    εpos : LT.lt 0 ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    hs : ∀ (p : GromovHausdorff.GHSpace), And (s p).Finite (HasSubset.Subset Set.u …
    N✝ : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → t.Finite → Nat
    e : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → (a : t.Finite) → Equiv ( …
    x✝ : ∀ (p : GromovHausdorff.GHSpace) (t : Set p.Rep), t.Finite → True
    N : GromovHausdorff.GHSpace → Nat := fun p => N✝ p (s p) ⋯
    E : (p : GromovHausdorff.GHSpace) → Equiv (↑(s p)) (Fin (N✝ p (s p) ⋯)) := fun …
    F : GromovHausdorff.GHSpace → Sigma fun n => Fin n → Fin n → Int := fun p => ⟨ …
    p q : GromovHausdorff.GHSpace
    hpq : Eq (F p) (F q)
    ⊢ LE.le (Dist.dist p q) δ
  -/
  have Npq : N p = N q := (Sigma.mk.inj_iff.1 hpq).1
  /-
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (2 / 5) δ
    εpos : LT.lt 0 ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    hs : ∀ (p : GromovHausdorff.GHSpace), And (s p).Finite (HasSubset.Subset Set.u …
    N✝ : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → t.Finite → Nat
    e : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → (a : t.Finite) → Equiv ( …
    x✝ : ∀ (p : GromovHausdorff.GHSpace) (t : Set p.Rep), t.Finite → True
    N : GromovHausdorff.GHSpace → Nat := fun p => N✝ p (s p) ⋯
    E : (p : GromovHausdorff.GHSpace) → Equiv (↑(s p)) (Fin (N✝ p (s p) ⋯)) := fun …
    F : GromovHausdorff.GHSpace → Sigma fun n => Fin n → Fin n → Int := fun p => ⟨ …
    p q : GromovHausdorff.GHSpace
    hpq : Eq (F p) (F q)
    Npq : Eq (N p) (N q)
    ⊢ LE.le (Dist.dist p q) δ
  -/
  let Ψ : s p → s q := fun x => (E q).symm (Fin.cast Npq ((E p) x))
  /-
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (2 / 5) δ
    εpos : LT.lt 0 ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    hs : ∀ (p : GromovHausdorff.GHSpace), And (s p).Finite (HasSubset.Subset Set.u …
    N✝ : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → t.Finite → Nat
    e : (p : GromovHausdorff.GHSpace) → (t : Set p.Rep) → (a : t.Finite) → Equiv ( …
    x✝ : ∀ (p : GromovHausdorff.GHSpace) (t : Set p.Rep), t.Finite → True
    N : GromovHausdorff.GHSpace → Nat := fun p => N✝ p (s p) ⋯
    E : (p : GromovHausdorff.GHSpace) → Equiv (↑(s p)) (Fin (N✝ p (s p) ⋯)) := fun …
    F : GromovHausdorff.GHSpace → Sigma fun n => Fin n → Fin n → Int := fun p => ⟨ …
    p q : GromovHausdorff.GHSpace
    hpq : Eq (F p) (F q)
    Npq : Eq (N p) (N q)
    Ψ : ↑(s p) → ↑(s q) := fun x => (E q).symm (Fin.cast Npq ((E p) x))
    ⊢ LE.le (Dist.dist p q) δ
  -/
  let Φ : s p → q.Rep := fun x => Ψ x
  -- Use the almost isometry `Φ` to show that `p.rep` and `q.rep`
  -- are within controlled Gromov-Hausdorff distance.
  have main : ghDist p.Rep q.Rep ≤ ε + ε / 2 + ε := by
    refine ghDist_le_of_approx_subsets Φ ?_ ?_ ?_
    · show ∀ x : p.Rep, ∃ y ∈ s p, dist x y ≤ ε
      -- by construction, `s p` is `ε`-dense
      intro x
      have : x ∈ ⋃ y ∈ s p, ball y ε := (hs p).2 (mem_univ _)
      rcases mem_iUnion₂.1 this with ⟨y, ys, hy⟩
      exact ⟨y, ys, le_of_lt hy⟩
    · show ∀ x : q.Rep, ∃ z : s p, dist x (Φ z) ≤ ε
      -- by construction, `s q` is `ε`-dense, and it is the range of `Φ`
      intro x
      have : x ∈ ⋃ y ∈ s q, ball y ε := (hs q).2 (mem_univ _)
      rcases mem_iUnion₂.1 this with ⟨y, ys, hy⟩
      let i : ℕ := E q ⟨y, ys⟩
      let hi := ((E q) ⟨y, ys⟩).is_lt
      have ihi_eq : (⟨i, hi⟩ : Fin (N q)) = (E q) ⟨y, ys⟩ := by rw [Fin.ext_iff, Fin.val_mk]
      have hiq : i < N q := hi
      have hip : i < N p := by rwa [Npq.symm] at hiq
      let z := (E p).symm ⟨i, hip⟩
      use z
      have C1 : (E p) z = ⟨i, hip⟩ := (E p).apply_symm_apply ⟨i, hip⟩
      have C2 : Fin.cast Npq ⟨i, hip⟩ = ⟨i, hi⟩ := rfl
      have C3 : (E q).symm ⟨i, hi⟩ = ⟨y, ys⟩ := by
        rw [ihi_eq]; exact (E q).symm_apply_apply ⟨y, ys⟩
      have : Φ z = y := by simp only [Φ, Ψ]; rw [C1, C2, C3]
      rw [this]
      exact le_of_lt hy
    · show ∀ x y : s p, |dist x y - dist (Φ x) (Φ y)| ≤ ε
      /- the distance between `x` and `y` is encoded in `F p`, and the distance between
            `Φ x` and `Φ y` (two points of `s q`) is encoded in `F q`, all this up to `ε`.
            As `F p = F q`, the distances are almost equal. -/
      -- Porting note: we have to circumvent the absence of `change … with … `
      intro x y
      -- have : dist (Φ x) (Φ y) = dist (Ψ x) (Ψ y) := rfl
      rw [show dist (Φ x) (Φ y) = dist (Ψ x) (Ψ y) from rfl]
      -- introduce `i`, that codes both `x` and `Φ x` in `Fin (N p) = Fin (N q)`
      let i : ℕ := E p x
      have hip : i < N p := ((E p) x).2
      have hiq : i < N q := by rwa [Npq] at hip
      have i' : i = (E q) (Ψ x) := by simp only [i, Ψ, Equiv.apply_symm_apply, Fin.coe_cast]
      -- introduce `j`, that codes both `y` and `Φ y` in `Fin (N p) = Fin (N q)`
      let j : ℕ := E p y
      have hjp : j < N p := ((E p) y).2
      have hjq : j < N q := by rwa [Npq] at hjp
      have j' : j = ((E q) (Ψ y)).1 := by
        simp only [j, Ψ, Equiv.apply_symm_apply, Fin.coe_cast]
      -- Express `dist x y` in terms of `F p`
      have : (F p).2 ((E p) x) ((E p) y) = ⌊ε⁻¹ * dist x y⌋ := by
        simp only [F, (E p).symm_apply_apply]
      have Ap : (F p).2 ⟨i, hip⟩ ⟨j, hjp⟩ = ⌊ε⁻¹ * dist x y⌋ := by rw [← this]
      -- Express `dist (Φ x) (Φ y)` in terms of `F q`
      have : (F q).2 ((E q) (Ψ x)) ((E q) (Ψ y)) = ⌊ε⁻¹ * dist (Ψ x) (Ψ y)⌋ := by
        simp only [F, (E q).symm_apply_apply]
      have Aq : (F q).2 ⟨i, hiq⟩ ⟨j, hjq⟩ = ⌊ε⁻¹ * dist (Ψ x) (Ψ y)⌋ := by
        rw [← this]
        -- Porting note: `congr` fails to make progress
        refine congr_arg₂ (F q).2 ?_ ?_ <;> ext1
        exacts [i', j']
      -- use the equality between `F p` and `F q` to deduce that the distances have equal
      -- integer parts
      have : (F p).2 ⟨i, hip⟩ ⟨j, hjp⟩ = (F q).2 ⟨i, hiq⟩ ⟨j, hjq⟩ := by
        have hpq' : HEq (F p).snd (F q).snd := (Sigma.mk.inj_iff.1 hpq).2
        rw [Fin.heq_fun₂_iff Npq Npq] at hpq'
        rw [← hpq']
        -- Porting note: new version above, because `change … with…` is not implemented
        -- we want to `subst hpq` where `hpq : F p = F q`, except that `subst` only works
        -- with a constant, so replace `F q` (and everything that depends on it) by a constant `f`
        -- then `subst`
        -- revert hiq hjq
        -- change N q with (F q).1
        -- generalize F q = f at hpq ⊢
        -- subst hpq
        -- rfl
      rw [Ap, Aq] at this
      -- deduce that the distances coincide up to `ε`, by a straightforward computation
      -- that should be automated
      have I :=
        calc
          |ε⁻¹| * |dist x y - dist (Ψ x) (Ψ y)| = |ε⁻¹ * (dist x y - dist (Ψ x) (Ψ y))| :=
            (abs_mul _ _).symm
          _ = |ε⁻¹ * dist x y - ε⁻¹ * dist (Ψ x) (Ψ y)| := by congr; ring
          _ ≤ 1 := le_of_lt (abs_sub_lt_one_of_floor_eq_floor this)
      calc
        |dist x y - dist (Ψ x) (Ψ y)| = ε * ε⁻¹ * |dist x y - dist (Ψ x) (Ψ y)| := by
          rw [mul_inv_cancel₀ (ne_of_gt εpos), one_mul]
        _ = ε * (|ε⁻¹| * |dist x y - dist (Ψ x) (Ψ y)|) := by
          rw [abs_of_nonneg (le_of_lt (inv_pos.2 εpos)), mul_assoc]
        _ ≤ ε * 1 := mul_le_mul_of_nonneg_left I (le_of_lt εpos)
        _ = ε := mul_one _
  calc
    dist p q = ghDist p.Rep q.Rep := dist_ghDist p q
    _ ≤ ε + ε / 2 + ε := main
    _ = δ := by ring


/-- Compactness criterion: a closed set of compact metric spaces is compact if the spaces have
a uniformly bounded diameter, and for all `ε` the number of balls of radius `ε` required
to cover the spaces is uniformly bounded. This is an equivalence, but we only prove the
interesting direction that these conditions imply compactness. -/
theorem totallyBounded {t : Set GHSpace} {C : ℝ} {u : ℕ → ℝ} {K : ℕ → ℕ}
    (ulim : Tendsto u atTop (𝓝 0)) (hdiam : ∀ p ∈ t, diam (univ : Set (GHSpace.Rep p)) ≤ C)
    (hcov : ∀ p ∈ t, ∀ n : ℕ, ∃ s : Set (GHSpace.Rep p),
      (#s) ≤ K n ∧ univ ⊆ ⋃ x ∈ s, ball x (u n)) :
    TotallyBounded t := by
  /- Let `δ>0`, and `ε = δ/5`. For each `p`, we construct a finite subset `s p` of `p`, which
    is `ε`-dense and has cardinality at most `K n`. Encoding the mutual distances of points
    in `s p`, up to `ε`, we will get a map `F` associating to `p` finitely many data, and making
    it possible to reconstruct `p` up to `ε`. This is enough to prove total boundedness. -/
  /-
    t : Set GromovHausdorff.GHSpace
    C : Real
    u : Nat → Real
    K : Nat → Nat
    ulim : Filter.Tendsto u Filter.atTop (nhds 0)
    hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
    hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
    ⊢ TotallyBounded t
  -/
  refine Metric.totallyBounded_of_finite_discretization fun δ δpos => ?_
  /-
    t : Set GromovHausdorff.GHSpace
    C : Real
    u : Nat → Real
    K : Nat → Nat
    ulim : Filter.Tendsto u Filter.atTop (nhds 0)
    hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
    hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
    δ : Real
    δpos : GT.gt δ 0
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : ↑t), Eq (F x) (F y) …
  -/
  let ε := 1 / 5 * δ
  /-
    t : Set GromovHausdorff.GHSpace
    C : Real
    u : Nat → Real
    K : Nat → Nat
    ulim : Filter.Tendsto u Filter.atTop (nhds 0)
    hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
    hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (1 / 5) δ
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : ↑t), Eq (F x) (F y) …
  -/
  have εpos : 0 < ε := mul_pos (by norm_num) δpos
  -- choose `n` for which `u n < ε`
  /-
    t : Set GromovHausdorff.GHSpace
    C : Real
    u : Nat → Real
    K : Nat → Nat
    ulim : Filter.Tendsto u Filter.atTop (nhds 0)
    hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
    hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (1 / 5) δ
    εpos : LT.lt 0 ε
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : ↑t), Eq (F x) (F y) …
  -/
  rcases Metric.tendsto_atTop.1 ulim ε εpos with ⟨n, hn⟩
  have u_le_ε : u n ≤ ε := by
    have := hn n le_rfl
    simp only [Real.dist_eq, add_zero, sub_eq_add_neg, neg_zero] at this
    exact le_of_lt (lt_of_le_of_lt (le_abs_self _) this)
  -- construct a finite subset `s p` of `p` which is `ε`-dense and has cardinal `≤ K n`
  have :
    ∀ p : GHSpace,
      ∃ s : Set p.Rep, ∃ N ≤ K n, ∃ _ : Equiv s (Fin N), p ∈ t → univ ⊆ ⋃ x ∈ s, ball x (u n) := by
    intro p
    by_cases hp : p ∉ t
    · have : Nonempty (Equiv (∅ : Set p.Rep) (Fin 0)) := by
        rw [← Fintype.card_eq]
        simp only [empty_card', Fintype.card_fin]
      use ∅, 0, bot_le, this.some
      -- Porting note: unclear why this next line wasn't needed in Lean 3
      exact fun hp' => (hp hp').elim
    · rcases hcov _ (Set.not_not_mem.1 hp) n with ⟨s, ⟨scard, scover⟩⟩
      rcases Cardinal.lt_aleph0.1 (lt_of_le_of_lt scard (Cardinal.nat_lt_aleph0 _)) with ⟨N, hN⟩
      rw [hN, Nat.cast_le] at scard
      have : #s = #(Fin N) := by rw [hN, Cardinal.mk_fin]
      cases' Quotient.exact this with E
      use s, N, scard, E
      simp only [scover, imp_true_iff]
  /-
    case intro
    t : Set GromovHausdorff.GHSpace
    C : Real
    u : Nat → Real
    K : Nat → Nat
    ulim : Filter.Tendsto u Filter.atTop (nhds 0)
    hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
    hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (1 / 5) δ
    εpos : LT.lt 0 ε
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LT.lt (Dist.dist (u n_1) 0) ε
    u_le_ε : LE.le (u n) ε
    this : ∀ (p : GromovHausdorff.GHSpace), Exists fun s => Exists fun N => And (L …
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : ↑t), Eq (F x) (F y) …
  -/
  choose s N hN E hs using this
  -- Define a function `F` taking values in a finite type and associating to `p` enough data
  -- to reconstruct it up to `ε`, namely the (discretized) distances between elements of `s p`.
  /-
    case intro
    t : Set GromovHausdorff.GHSpace
    C : Real
    u : Nat → Real
    K : Nat → Nat
    ulim : Filter.Tendsto u Filter.atTop (nhds 0)
    hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
    hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (1 / 5) δ
    εpos : LT.lt 0 ε
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LT.lt (Dist.dist (u n_1) 0) ε
    u_le_ε : LE.le (u n) ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    N : GromovHausdorff.GHSpace → Nat
    hN : ∀ (p : GromovHausdorff.GHSpace), LE.le (N p) (K n)
    E : (p : GromovHausdorff.GHSpace) → Equiv (↑(s p)) (Fin (N p))
    hs : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → HasSubset.Subset Se …
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : ↑t), Eq (F x) (F y) …
  -/
  let M := ⌊ε⁻¹ * max C 0⌋₊
  let F : GHSpace → Σ k : Fin (K n).succ, Fin k → Fin k → Fin M.succ := fun p =>
    ⟨⟨N p, lt_of_le_of_lt (hN p) (Nat.lt_succ_self _)⟩, fun a b =>
      ⟨min M ⌊ε⁻¹ * dist ((E p).symm a) ((E p).symm b)⌋₊,
        (min_le_left _ _).trans_lt (Nat.lt_succ_self _)⟩⟩
  /-
    case intro
    t : Set GromovHausdorff.GHSpace
    C : Real
    u : Nat → Real
    K : Nat → Nat
    ulim : Filter.Tendsto u Filter.atTop (nhds 0)
    hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
    hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (1 / 5) δ
    εpos : LT.lt 0 ε
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LT.lt (Dist.dist (u n_1) 0) ε
    u_le_ε : LE.le (u n) ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    N : GromovHausdorff.GHSpace → Nat
    hN : ∀ (p : GromovHausdorff.GHSpace), LE.le (N p) (K n)
    E : (p : GromovHausdorff.GHSpace) → Equiv (↑(s p)) (Fin (N p))
    hs : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → HasSubset.Subset Se …
    M : Nat := Nat.floor (HMul.hMul (Inv.inv ε) (Max.max C 0))
    F : GromovHausdorff.GHSpace → Sigma fun k => Fin ↑k → Fin ↑k → Fin M.succ := f …
    ⊢ Exists fun β => Exists fun x => Exists fun F => ∀ (x y : ↑t), Eq (F x) (F y) …
  -/
  refine ⟨_, ?_, fun p => F p, ?_⟩
    /-
      case intro.refine_1
      t : Set GromovHausdorff.GHSpace
      C : Real
      u : Nat → Real
      K : Nat → Nat
      ulim : Filter.Tendsto u Filter.atTop (nhds 0)
      hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
      hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
      δ : Real
      δpos : GT.gt δ 0
      ε : Real := HMul.hMul (1 / 5) δ
      εpos : LT.lt 0 ε
      n : Nat
      hn : ∀ (n_1 : Nat), GE.ge n_1 n → LT.lt (Dist.dist (u n_1) 0) ε
      u_le_ε : LE.le (u n) ε
      s : (p : GromovHausdorff.GHSpace) → Set p.Rep
      N : GromovHausdorff.GHSpace → Nat
      hN : ∀ (p : GromovHausdorff.GHSpace), LE.le (N p) (K n)
      E : (p : GromovHausdorff.GHSpace) → Equiv (↑(s p)) (Fin (N p))
      hs : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → HasSubset.Subset Se …
      M : Nat := Nat.floor (HMul.hMul (Inv.inv ε) (Max.max C 0))
      F : GromovHausdorff.GHSpace → Sigma fun k => Fin ↑k → Fin ↑k → Fin M.succ := f …
      ⊢ Fintype (Sigma fun k => Fin ↑k → Fin ↑k → Fin M.succ)
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
  -- It remains to show that if `F p = F q`, then `p` and `q` are `ε`-close
  /-
    case intro.refine_2
    t : Set GromovHausdorff.GHSpace
    C : Real
    u : Nat → Real
    K : Nat → Nat
    ulim : Filter.Tendsto u Filter.atTop (nhds 0)
    hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
    hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (1 / 5) δ
    εpos : LT.lt 0 ε
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LT.lt (Dist.dist (u n_1) 0) ε
    u_le_ε : LE.le (u n) ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    N : GromovHausdorff.GHSpace → Nat
    hN : ∀ (p : GromovHausdorff.GHSpace), LE.le (N p) (K n)
    E : (p : GromovHausdorff.GHSpace) → Equiv (↑(s p)) (Fin (N p))
    hs : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → HasSubset.Subset Se …
    M : Nat := Nat.floor (HMul.hMul (Inv.inv ε) (Max.max C 0))
    F : GromovHausdorff.GHSpace → Sigma fun k => Fin ↑k → Fin ↑k → Fin M.succ := f …
    ⊢ ∀ (x y : ↑t), Eq ((fun p => F ↑p) x) ((fun p => F ↑p) y) → LT.lt (Dist.dist  …
  -/
  rintro ⟨p, pt⟩ ⟨q, qt⟩ hpq
  /-
    case intro.refine_2.mk.mk
    t : Set GromovHausdorff.GHSpace
    C : Real
    u : Nat → Real
    K : Nat → Nat
    ulim : Filter.Tendsto u Filter.atTop (nhds 0)
    hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
    hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (1 / 5) δ
    εpos : LT.lt 0 ε
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LT.lt (Dist.dist (u n_1) 0) ε
    u_le_ε : LE.le (u n) ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    N : GromovHausdorff.GHSpace → Nat
    hN : ∀ (p : GromovHausdorff.GHSpace), LE.le (N p) (K n)
    E : (p : GromovHausdorff.GHSpace) → Equiv (↑(s p)) (Fin (N p))
    hs : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → HasSubset.Subset Se …
    M : Nat := Nat.floor (HMul.hMul (Inv.inv ε) (Max.max C 0))
    F : GromovHausdorff.GHSpace → Sigma fun k => Fin ↑k → Fin ↑k → Fin M.succ := f …
    p : GromovHausdorff.GHSpace
    pt : Membership.mem t p
    q : GromovHausdorff.GHSpace
    qt : Membership.mem t q
    hpq : Eq ((fun p => F ↑p) ⟨p, pt⟩) ((fun p => F ↑p) ⟨q, qt⟩)
    ⊢ LT.lt (Dist.dist ↑⟨p, pt⟩ ↑⟨q, qt⟩) δ
  -/
  have Npq : N p = N q := Fin.ext_iff.1 (Sigma.mk.inj_iff.1 hpq).1
  /-
    case intro.refine_2.mk.mk
    t : Set GromovHausdorff.GHSpace
    C : Real
    u : Nat → Real
    K : Nat → Nat
    ulim : Filter.Tendsto u Filter.atTop (nhds 0)
    hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
    hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (1 / 5) δ
    εpos : LT.lt 0 ε
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LT.lt (Dist.dist (u n_1) 0) ε
    u_le_ε : LE.le (u n) ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    N : GromovHausdorff.GHSpace → Nat
    hN : ∀ (p : GromovHausdorff.GHSpace), LE.le (N p) (K n)
    E : (p : GromovHausdorff.GHSpace) → Equiv (↑(s p)) (Fin (N p))
    hs : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → HasSubset.Subset Se …
    M : Nat := Nat.floor (HMul.hMul (Inv.inv ε) (Max.max C 0))
    F : GromovHausdorff.GHSpace → Sigma fun k => Fin ↑k → Fin ↑k → Fin M.succ := f …
    p : GromovHausdorff.GHSpace
    pt : Membership.mem t p
    q : GromovHausdorff.GHSpace
    qt : Membership.mem t q
    hpq : Eq ((fun p => F ↑p) ⟨p, pt⟩) ((fun p => F ↑p) ⟨q, qt⟩)
    Npq : Eq (N p) (N q)
    ⊢ LT.lt (Dist.dist ↑⟨p, pt⟩ ↑⟨q, qt⟩) δ
  -/
  let Ψ : s p → s q := fun x => (E q).symm (Fin.cast Npq ((E p) x))
  /-
    case intro.refine_2.mk.mk
    t : Set GromovHausdorff.GHSpace
    C : Real
    u : Nat → Real
    K : Nat → Nat
    ulim : Filter.Tendsto u Filter.atTop (nhds 0)
    hdiam : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → LE.le (Metric.di …
    hcov : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → ∀ (n : Nat), Exis …
    δ : Real
    δpos : GT.gt δ 0
    ε : Real := HMul.hMul (1 / 5) δ
    εpos : LT.lt 0 ε
    n : Nat
    hn : ∀ (n_1 : Nat), GE.ge n_1 n → LT.lt (Dist.dist (u n_1) 0) ε
    u_le_ε : LE.le (u n) ε
    s : (p : GromovHausdorff.GHSpace) → Set p.Rep
    N : GromovHausdorff.GHSpace → Nat
    hN : ∀ (p : GromovHausdorff.GHSpace), LE.le (N p) (K n)
    E : (p : GromovHausdorff.GHSpace) → Equiv (↑(s p)) (Fin (N p))
    hs : ∀ (p : GromovHausdorff.GHSpace), Membership.mem t p → HasSubset.Subset Se …
    M : Nat := Nat.floor (HMul.hMul (Inv.inv ε) (Max.max C 0))
    F : GromovHausdorff.GHSpace → Sigma fun k => Fin ↑k → Fin ↑k → Fin M.succ := f …
    p : GromovHausdorff.GHSpace
    pt : Membership.mem t p
    q : GromovHausdorff.GHSpace
    qt : Membership.mem t q
    hpq : Eq ((fun p => F ↑p) ⟨p, pt⟩) ((fun p => F ↑p) ⟨q, qt⟩)
    Npq : Eq (N p) (N q)
    Ψ : ↑(s p) → ↑(s q) := fun x => (E q).symm (Fin.cast Npq ((E p) x))
    ⊢ LT.lt (Dist.dist ↑⟨p, pt⟩ ↑⟨q, qt⟩) δ
  -/
  let Φ : s p → q.Rep := fun x => Ψ x
  have main : ghDist p.Rep q.Rep ≤ ε + ε / 2 + ε := by
    -- to prove the main inequality, argue that `s p` is `ε`-dense in `p`, and `s q` is `ε`-dense
    -- in `q`, and `s p` and `s q` are almost isometric. Then closeness follows
    -- from `ghDist_le_of_approx_subsets`
    refine ghDist_le_of_approx_subsets Φ ?_ ?_ ?_
    · show ∀ x : p.Rep, ∃ y ∈ s p, dist x y ≤ ε
      -- by construction, `s p` is `ε`-dense
      intro x
      have : x ∈ ⋃ y ∈ s p, ball y (u n) := (hs p pt) (mem_univ _)
      rcases mem_iUnion₂.1 this with ⟨y, ys, hy⟩
      exact ⟨y, ys, le_trans (le_of_lt hy) u_le_ε⟩
    · show ∀ x : q.Rep, ∃ z : s p, dist x (Φ z) ≤ ε
      -- by construction, `s q` is `ε`-dense, and it is the range of `Φ`
      intro x
      have : x ∈ ⋃ y ∈ s q, ball y (u n) := (hs q qt) (mem_univ _)
      rcases mem_iUnion₂.1 this with ⟨y, ys, hy⟩
      let i : ℕ := E q ⟨y, ys⟩
      let hi := ((E q) ⟨y, ys⟩).2
      have ihi_eq : (⟨i, hi⟩ : Fin (N q)) = (E q) ⟨y, ys⟩ := by rw [Fin.ext_iff, Fin.val_mk]
      have hiq : i < N q := hi
      have hip : i < N p := by rwa [Npq.symm] at hiq
      let z := (E p).symm ⟨i, hip⟩
      use z
      have C1 : (E p) z = ⟨i, hip⟩ := (E p).apply_symm_apply ⟨i, hip⟩
      have C2 : Fin.cast Npq ⟨i, hip⟩ = ⟨i, hi⟩ := rfl
      have C3 : (E q).symm ⟨i, hi⟩ = ⟨y, ys⟩ := by
        rw [ihi_eq]; exact (E q).symm_apply_apply ⟨y, ys⟩
      have : Φ z = y := by simp only [Ψ, Φ]; rw [C1, C2, C3]
      rw [this]
      exact le_trans (le_of_lt hy) u_le_ε
    · show ∀ x y : s p, |dist x y - dist (Φ x) (Φ y)| ≤ ε
      /- the distance between `x` and `y` is encoded in `F p`, and the distance between
            `Φ x` and `Φ y` (two points of `s q`) is encoded in `F q`, all this up to `ε`.
            As `F p = F q`, the distances are almost equal. -/
      intro x y
      have : dist (Φ x) (Φ y) = dist (Ψ x) (Ψ y) := rfl
      rw [this]
      -- introduce `i`, that codes both `x` and `Φ x` in `Fin (N p) = Fin (N q)`
      let i : ℕ := E p x
      have hip : i < N p := ((E p) x).2
      have hiq : i < N q := by rwa [Npq] at hip
      have i' : i = (E q) (Ψ x) := by simp only [i, Ψ, Equiv.apply_symm_apply, Fin.coe_cast]
      -- introduce `j`, that codes both `y` and `Φ y` in `Fin (N p) = Fin (N q)`
      let j : ℕ := E p y
      have hjp : j < N p := ((E p) y).2
      have hjq : j < N q := by rwa [Npq] at hjp
      have j' : j = (E q) (Ψ y) := by simp only [j, Ψ, Equiv.apply_symm_apply, Fin.coe_cast]
      -- Express `dist x y` in terms of `F p`
      have Ap : ((F p).2 ⟨i, hip⟩ ⟨j, hjp⟩).1 = ⌊ε⁻¹ * dist x y⌋₊ :=
        calc
          ((F p).2 ⟨i, hip⟩ ⟨j, hjp⟩).1 = ((F p).2 ((E p) x) ((E p) y)).1 := by
            congr
          _ = min M ⌊ε⁻¹ * dist x y⌋₊ := by simp only [F, (E p).symm_apply_apply]
          _ = ⌊ε⁻¹ * dist x y⌋₊ := by
            refine min_eq_right (Nat.floor_mono ?_)
            refine mul_le_mul_of_nonneg_left (le_trans ?_ (le_max_left _ _)) (inv_pos.2 εpos).le
            change dist (x : p.Rep) y ≤ C
            refine (dist_le_diam_of_mem isCompact_univ.isBounded (mem_univ _) (mem_univ _)).trans ?_
            exact hdiam p pt
      -- Express `dist (Φ x) (Φ y)` in terms of `F q`
      have Aq : ((F q).2 ⟨i, hiq⟩ ⟨j, hjq⟩).1 = ⌊ε⁻¹ * dist (Ψ x) (Ψ y)⌋₊ :=
        calc
          ((F q).2 ⟨i, hiq⟩ ⟨j, hjq⟩).1 = ((F q).2 ((E q) (Ψ x)) ((E q) (Ψ y))).1 := by
            -- Porting note: `congr` drops `Fin.val` but fails to make further progress
            exact congr_arg₂ (Fin.val <| (F q).2 · ·) (Fin.ext i') (Fin.ext j')
          _ = min M ⌊ε⁻¹ * dist (Ψ x) (Ψ y)⌋₊ := by simp only [F, (E q).symm_apply_apply]
          _ = ⌊ε⁻¹ * dist (Ψ x) (Ψ y)⌋₊ := by
            refine min_eq_right (Nat.floor_mono ?_)
            refine mul_le_mul_of_nonneg_left (le_trans ?_ (le_max_left _ _)) (inv_pos.2 εpos).le
            change dist (Ψ x : q.Rep) (Ψ y) ≤ C
            refine (dist_le_diam_of_mem isCompact_univ.isBounded (mem_univ _) (mem_univ _)).trans ?_
            exact hdiam q qt
      -- use the equality between `F p` and `F q` to deduce that the distances have equal
      -- integer parts
      have : ((F p).2 ⟨i, hip⟩ ⟨j, hjp⟩).1 = ((F q).2 ⟨i, hiq⟩ ⟨j, hjq⟩).1 := by
        have hpq' : HEq (F p).snd (F q).snd := (Sigma.mk.inj_iff.1 hpq).2
        rw [Fin.heq_fun₂_iff Npq Npq] at hpq'
        rw [← hpq']
        -- Porting note: new version above because `subst…` does not work
        -- we want to `subst hpq` where `hpq : F p = F q`, except that `subst` only works
        -- with a constant, so replace `F q` (and everything that depends on it) by a constant `f`
        -- then `subst`
        -- dsimp only [show N q = (F q).1 from rfl] at hiq hjq ⊢
        -- generalize F q = f at hpq ⊢
        -- subst hpq
        -- intros
        -- rfl
      have : ⌊ε⁻¹ * dist x y⌋ = ⌊ε⁻¹ * dist (Ψ x) (Ψ y)⌋ := by
        rw [Ap, Aq] at this
        have D : 0 ≤ ⌊ε⁻¹ * dist x y⌋ :=
          floor_nonneg.2 (mul_nonneg (le_of_lt (inv_pos.2 εpos)) dist_nonneg)
        have D' : 0 ≤ ⌊ε⁻¹ * dist (Ψ x) (Ψ y)⌋ :=
          floor_nonneg.2 (mul_nonneg (le_of_lt (inv_pos.2 εpos)) dist_nonneg)
        rw [← Int.toNat_of_nonneg D, ← Int.toNat_of_nonneg D', Int.floor_toNat, Int.floor_toNat,
          this]
      -- deduce that the distances coincide up to `ε`, by a straightforward computation
      -- that should be automated
      have I :=
        calc
          |ε⁻¹| * |dist x y - dist (Ψ x) (Ψ y)| = |ε⁻¹ * (dist x y - dist (Ψ x) (Ψ y))| :=
            (abs_mul _ _).symm
          _ = |ε⁻¹ * dist x y - ε⁻¹ * dist (Ψ x) (Ψ y)| := by congr; ring
          _ ≤ 1 := le_of_lt (abs_sub_lt_one_of_floor_eq_floor this)
      calc
        |dist x y - dist (Ψ x) (Ψ y)| = ε * ε⁻¹ * |dist x y - dist (Ψ x) (Ψ y)| := by
          rw [mul_inv_cancel₀ (ne_of_gt εpos), one_mul]
        _ = ε * (|ε⁻¹| * |dist x y - dist (Ψ x) (Ψ y)|) := by
          rw [abs_of_nonneg (le_of_lt (inv_pos.2 εpos)), mul_assoc]
        _ ≤ ε * 1 := mul_le_mul_of_nonneg_left I (le_of_lt εpos)
        _ = ε := mul_one _
  calc
    dist p q = ghDist p.Rep q.Rep := dist_ghDist p q
    _ ≤ ε + ε / 2 + ε := main
    _ = δ / 2 := by simp only [ε, one_div]; ring
    _ < δ := half_lt_self δpos


/-- Auxiliary structure used to glue metric spaces below, recording an isometric embedding
of a type `A` in another metric space. -/
structure AuxGluingStruct (A : Type) [MetricSpace A] : Type 1 where
  Space : Type
  metric : MetricSpace Space
  embed : A → Space
  isom : Isometry embed


instance (A : Type) [MetricSpace A] : Inhabited (AuxGluingStruct A) :=
  ⟨{  Space := A
                   /-
                     X : Nat → Type
                     inst✝³ : (n : Nat) → MetricSpace (X n)
                     inst✝² : ∀ (n : Nat), CompactSpace (X n)
                     inst✝¹ : ∀ (n : Nat), Nonempty (X n)
                     A : Type
                     inst✝ : MetricSpace A
                     ⊢ MetricSpace A
                   -/
      metric := by infer_instance
                   /-
                     🎉 no goals
                   -/
      embed := id
      -- Porting note: without `by exact` there was an unsolved metavariable
                            /-
                              X : Nat → Type
                              inst✝³ : (n : Nat) → MetricSpace (X n)
                              inst✝² : ∀ (n : Nat), CompactSpace (X n)
                              inst✝¹ : ∀ (n : Nat), Nonempty (X n)
                              A : Type
                              inst✝ : MetricSpace A
                              x y : A
                              ⊢ Eq (EDist.edist (id x) (id y)) (EDist.edist x y)
                            -/
      isom := fun x y => by exact rfl }⟩
                            /-
                              🎉 no goals
                            -/


/-- Auxiliary sequence of metric spaces, containing copies of `X 0`, ..., `X n`, where each
`X i` is glued to `X (i+1)` in an optimal way. The space at step `n+1` is obtained from the space
at step `n` by adding `X (n+1)`, glued in an optimal way to the `X n` already sitting there. -/
def auxGluing (n : ℕ) : AuxGluingStruct (X n) :=
  Nat.recOn n default fun n Y =>
    { Space := GlueSpace Y.isom (isometry_optimalGHInjl (X n) (X (n + 1)))
                   /-
                     X : Nat → Type
                     inst✝² : (n : Nat) → MetricSpace (X n)
                     inst✝¹ : ∀ (n : Nat), CompactSpace (X n)
                     inst✝ : ∀ (n : Nat), Nonempty (X n)
                     n✝ n : Nat
                     Y : GromovHausdorff.AuxGluingStruct (X n)
                     ⊢ MetricSpace (Metric.GlueSpace ⋯ ⋯)
                   -/
      metric := by infer_instance
                   /-
                     🎉 no goals
                   -/
      embed :=
        toGlueR Y.isom (isometry_optimalGHInjl (X n) (X (n + 1))) ∘ optimalGHInjr (X n) (X (n + 1))
      isom := (toGlueR_isometry _ _).comp (isometry_optimalGHInjr (X n) (X (n + 1))) }


/-- The Gromov-Hausdorff space is complete. -/
instance : CompleteSpace GHSpace := by
  /-
    X : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X n)
    inst✝ : ∀ (n : Nat), Nonempty (X n)
    ⊢ CompleteSpace GromovHausdorff.GHSpace
  -/
  set d := fun n : ℕ ↦ ((1 : ℝ) / 2) ^ n
  /-
    X : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X n)
    inst✝ : ∀ (n : Nat), Nonempty (X n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    ⊢ CompleteSpace GromovHausdorff.GHSpace
  -/
  have : ∀ n : ℕ, 0 < d n := fun _ ↦ by positivity
  -- start from a sequence of nonempty compact metric spaces within distance `1/2^n` of each other
  /-
    X : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X n)
    inst✝ : ∀ (n : Nat), Nonempty (X n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this : ∀ (n : Nat), LT.lt 0 (d n)
    ⊢ CompleteSpace GromovHausdorff.GHSpace
  -/
  refine Metric.complete_of_convergent_controlled_sequences d this fun u hu => ?_
  -- `X n` is a representative of `u n`
  /-
    X : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X n)
    inst✝ : ∀ (n : Nat), Nonempty (X n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  let X n := (u n).Rep
  -- glue them together successively in an optimal way, getting a sequence of metric spaces `Y n`
  /-
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  let Y := auxGluing X
  -- this equality is true by definition but Lean unfolds some defs in the wrong order
  have E :
    ∀ n : ℕ,
      GlueSpace (Y n).isom (isometry_optimalGHInjl (X n) (X (n + 1))) = (Y (n + 1)).Space :=
    fun n => by dsimp only [Y, auxGluing]
  /-
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  let c n := cast (E n)
  /-
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  have ic : ∀ n, Isometry (c n) := fun n x y => by dsimp only [Y, auxGluing]; exact rfl
  -- there is a canonical embedding of `Y n` in `Y (n+1)`, by construction
  let f : ∀ n, (Y n).Space → (Y (n + 1)).Space := fun n =>
    c n ∘ toGlueL (Y n).isom (isometry_optimalGHInjl (X n) (X n.succ))
  /-
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ic : ∀ (n : Nat), Isometry (c n)
    f : (n : Nat) → (Y n).Space → (Y (HAdd.hAdd n 1)).Space := fun n => Function.c …
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  have I : ∀ n, Isometry (f n) := fun n => (ic n).comp (toGlueL_isometry _ _)
  -- consider the inductive limit `Z0` of the `Y n`, and then its completion `Z`
  /-
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ic : ∀ (n : Nat), Isometry (c n)
    f : (n : Nat) → (Y n).Space → (Y (HAdd.hAdd n 1)).Space := fun n => Function.c …
    I : ∀ (n : Nat), Isometry (f n)
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  let Z0 := Metric.InductiveLimit I
  /-
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ic : ∀ (n : Nat), Isometry (c n)
    f : (n : Nat) → (Y n).Space → (Y (HAdd.hAdd n 1)).Space := fun n => Function.c …
    I : ∀ (n : Nat), Isometry (f n)
    Z0 : Type := Metric.InductiveLimit I
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  let Z := UniformSpace.Completion Z0
  /-
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ic : ∀ (n : Nat), Isometry (c n)
    f : (n : Nat) → (Y n).Space → (Y (HAdd.hAdd n 1)).Space := fun n => Function.c …
    I : ∀ (n : Nat), Isometry (f n)
    Z0 : Type := Metric.InductiveLimit I
    Z : Type := UniformSpace.Completion Z0
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  let Φ := toInductiveLimit I
  /-
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ic : ∀ (n : Nat), Isometry (c n)
    f : (n : Nat) → (Y n).Space → (Y (HAdd.hAdd n 1)).Space := fun n => Function.c …
    I : ∀ (n : Nat), Isometry (f n)
    Z0 : Type := Metric.InductiveLimit I
    Z : Type := UniformSpace.Completion Z0
    Φ : (n : Nat) → (Y n).Space → Metric.InductiveLimit I := Metric.toInductiveLim …
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  let coeZ := ((↑) : Z0 → Z)
  -- let `X2 n` be the image of `X n` in the space `Z`
  /-
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ic : ∀ (n : Nat), Isometry (c n)
    f : (n : Nat) → (Y n).Space → (Y (HAdd.hAdd n 1)).Space := fun n => Function.c …
    I : ∀ (n : Nat), Isometry (f n)
    Z0 : Type := Metric.InductiveLimit I
    Z : Type := UniformSpace.Completion Z0
    Φ : (n : Nat) → (Y n).Space → Metric.InductiveLimit I := Metric.toInductiveLim …
    coeZ : Z0 → UniformSpace.Completion Z0 := ↑Z0
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  let X2 n := range (coeZ ∘ Φ n ∘ (Y n).embed)
  have isom : ∀ n, Isometry (coeZ ∘ Φ n ∘ (Y n).embed) := by
    intro n
    refine UniformSpace.Completion.coe_isometry.comp ?_
    exact (toInductiveLimit_isometry _ _).comp (Y n).isom
  -- The Hausdorff distance of `X2 n` and `X2 (n+1)` is by construction the distance between
  -- `u n` and `u (n+1)`, therefore bounded by `1/2^n`
  have X2n : ∀ n, X2 n =
    range ((coeZ ∘ Φ n.succ ∘ c n ∘ toGlueR (Y n).isom
      (isometry_optimalGHInjl (X n) (X n.succ))) ∘ optimalGHInjl (X n) (X n.succ)) := by
    intro n
    change X2 n = range (coeZ ∘ Φ n.succ ∘ c n ∘
      toGlueR (Y n).isom (isometry_optimalGHInjl (X n) (X n.succ)) ∘
      optimalGHInjl (X n) (X n.succ))
    simp only [X2, Φ]
    rw [← toInductiveLimit_commute I]
    simp only [f]
    rw [← toGlue_commute]
    rfl
  -- simp_rw [range_comp] at X2n
  have X2nsucc : ∀ n, X2 n.succ =
      range ((coeZ ∘ Φ n.succ ∘ c n ∘ toGlueR (Y n).isom
        (isometry_optimalGHInjl (X n) (X n.succ))) ∘ optimalGHInjr (X n) (X n.succ)) := by
    intro n
    rfl
  -- simp_rw [range_comp] at X2nsucc
  have D2 : ∀ n, hausdorffDist (X2 n) (X2 n.succ) < d n := fun n ↦ by
    rw [X2n n, X2nsucc n, range_comp, range_comp, hausdorffDist_image,
      hausdorffDist_optimal, ← dist_ghDist]
    · exact hu n n n.succ (le_refl n) (le_succ n)
    · apply UniformSpace.Completion.coe_isometry.comp _
      exact (toInductiveLimit_isometry _ _).comp ((ic n).comp (toGlueR_isometry _ _))
  -- consider `X2 n` as a member `X3 n` of the type of nonempty compact subsets of `Z`, which
  -- is a metric space
  let X3 : ℕ → NonemptyCompacts Z := fun n =>
    ⟨⟨X2 n, isCompact_range (isom n).continuous⟩, range_nonempty _⟩
  -- `X3 n` is a Cauchy sequence by construction, as the successive distances are
  -- bounded by `(1/2)^n`
  have : CauchySeq X3 := by
    refine cauchySeq_of_le_geometric (1 / 2) 1 (by norm_num) fun n => ?_
    rw [one_mul]
    exact le_of_lt (D2 n)
  -- therefore, it converges to a limit `L`
  /-
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this✝ : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ic : ∀ (n : Nat), Isometry (c n)
    f : (n : Nat) → (Y n).Space → (Y (HAdd.hAdd n 1)).Space := fun n => Function.c …
    I : ∀ (n : Nat), Isometry (f n)
    Z0 : Type := Metric.InductiveLimit I
    Z : Type := UniformSpace.Completion Z0
    Φ : (n : Nat) → (Y n).Space → Metric.InductiveLimit I := Metric.toInductiveLim …
    coeZ : Z0 → UniformSpace.Completion Z0 := ↑Z0
    X2 : Nat → Set (UniformSpace.Completion Z0) := fun n => Set.range (Function.co …
    isom : ∀ (n : Nat), Isometry (Function.comp coeZ (Function.comp (Φ n) (Y n).em …
    X2n : ∀ (n : Nat), Eq (X2 n) (Set.range (Function.comp (Function.comp coeZ (Fu …
    X2nsucc : ∀ (n : Nat), Eq (X2 n.succ) (Set.range (Function.comp (Function.comp …
    D2 : ∀ (n : Nat), LT.lt (Metric.hausdorffDist (X2 n) (X2 n.succ)) (d n)
    X3 : Nat → TopologicalSpace.NonemptyCompacts Z := fun n => { carrier := X2 n,  …
    this : CauchySeq X3
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  rcases cauchySeq_tendsto_of_complete this with ⟨L, hL⟩
  -- By construction, the image of `X3 n` in the Gromov-Hausdorff space is `u n`.
  have : ∀ n, (NonemptyCompacts.toGHSpace ∘ X3) n = u n := by
    intro n
    rw [Function.comp_apply, NonemptyCompacts.toGHSpace, ← (u n).toGHSpace_rep,
      toGHSpace_eq_toGHSpace_iff_isometryEquiv]
    constructor
    convert (isom n).isometryEquivOnRange.symm
  -- the images of `X3 n` in the Gromov-Hausdorff space converge to the image of `L`
  -- so the images of `u n` converge to the image of `L` as well
  /-
    case intro
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this✝¹ : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ic : ∀ (n : Nat), Isometry (c n)
    f : (n : Nat) → (Y n).Space → (Y (HAdd.hAdd n 1)).Space := fun n => Function.c …
    I : ∀ (n : Nat), Isometry (f n)
    Z0 : Type := Metric.InductiveLimit I
    Z : Type := UniformSpace.Completion Z0
    Φ : (n : Nat) → (Y n).Space → Metric.InductiveLimit I := Metric.toInductiveLim …
    coeZ : Z0 → UniformSpace.Completion Z0 := ↑Z0
    X2 : Nat → Set (UniformSpace.Completion Z0) := fun n => Set.range (Function.co …
    isom : ∀ (n : Nat), Isometry (Function.comp coeZ (Function.comp (Φ n) (Y n).em …
    X2n : ∀ (n : Nat), Eq (X2 n) (Set.range (Function.comp (Function.comp coeZ (Fu …
    X2nsucc : ∀ (n : Nat), Eq (X2 n.succ) (Set.range (Function.comp (Function.comp …
    D2 : ∀ (n : Nat), LT.lt (Metric.hausdorffDist (X2 n) (X2 n.succ)) (d n)
    X3 : Nat → TopologicalSpace.NonemptyCompacts Z := fun n => { carrier := X2 n,  …
    this✝ : CauchySeq X3
    L : TopologicalSpace.NonemptyCompacts Z
    hL : Filter.Tendsto X3 Filter.atTop (nhds L)
    this : ∀ (n : Nat), Eq (Function.comp TopologicalSpace.NonemptyCompacts.toGHSp …
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  use L.toGHSpace
  /-
    case h
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this✝¹ : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ic : ∀ (n : Nat), Isometry (c n)
    f : (n : Nat) → (Y n).Space → (Y (HAdd.hAdd n 1)).Space := fun n => Function.c …
    I : ∀ (n : Nat), Isometry (f n)
    Z0 : Type := Metric.InductiveLimit I
    Z : Type := UniformSpace.Completion Z0
    Φ : (n : Nat) → (Y n).Space → Metric.InductiveLimit I := Metric.toInductiveLim …
    coeZ : Z0 → UniformSpace.Completion Z0 := ↑Z0
    X2 : Nat → Set (UniformSpace.Completion Z0) := fun n => Set.range (Function.co …
    isom : ∀ (n : Nat), Isometry (Function.comp coeZ (Function.comp (Φ n) (Y n).em …
    X2n : ∀ (n : Nat), Eq (X2 n) (Set.range (Function.comp (Function.comp coeZ (Fu …
    X2nsucc : ∀ (n : Nat), Eq (X2 n.succ) (Set.range (Function.comp (Function.comp …
    D2 : ∀ (n : Nat), LT.lt (Metric.hausdorffDist (X2 n) (X2 n.succ)) (d n)
    X3 : Nat → TopologicalSpace.NonemptyCompacts Z := fun n => { carrier := X2 n,  …
    this✝ : CauchySeq X3
    L : TopologicalSpace.NonemptyCompacts Z
    hL : Filter.Tendsto X3 Filter.atTop (nhds L)
    this : ∀ (n : Nat), Eq (Function.comp TopologicalSpace.NonemptyCompacts.toGHSp …
    ⊢ Filter.Tendsto u Filter.atTop (nhds L.toGHSpace)
  -/
  apply Filter.Tendsto.congr this
  /-
    case h
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this✝¹ : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ic : ∀ (n : Nat), Isometry (c n)
    f : (n : Nat) → (Y n).Space → (Y (HAdd.hAdd n 1)).Space := fun n => Function.c …
    I : ∀ (n : Nat), Isometry (f n)
    Z0 : Type := Metric.InductiveLimit I
    Z : Type := UniformSpace.Completion Z0
    Φ : (n : Nat) → (Y n).Space → Metric.InductiveLimit I := Metric.toInductiveLim …
    coeZ : Z0 → UniformSpace.Completion Z0 := ↑Z0
    X2 : Nat → Set (UniformSpace.Completion Z0) := fun n => Set.range (Function.co …
    isom : ∀ (n : Nat), Isometry (Function.comp coeZ (Function.comp (Φ n) (Y n).em …
    X2n : ∀ (n : Nat), Eq (X2 n) (Set.range (Function.comp (Function.comp coeZ (Fu …
    X2nsucc : ∀ (n : Nat), Eq (X2 n.succ) (Set.range (Function.comp (Function.comp …
    D2 : ∀ (n : Nat), LT.lt (Metric.hausdorffDist (X2 n) (X2 n.succ)) (d n)
    X3 : Nat → TopologicalSpace.NonemptyCompacts Z := fun n => { carrier := X2 n,  …
    this✝ : CauchySeq X3
    L : TopologicalSpace.NonemptyCompacts Z
    hL : Filter.Tendsto X3 Filter.atTop (nhds L)
    this : ∀ (n : Nat), Eq (Function.comp TopologicalSpace.NonemptyCompacts.toGHSp …
    ⊢ Filter.Tendsto (Function.comp TopologicalSpace.NonemptyCompacts.toGHSpace X3 …
  -/
  refine Tendsto.comp ?_ hL
  /-
    case h
    X✝ : Nat → Type
    inst✝² : (n : Nat) → MetricSpace (X✝ n)
    inst✝¹ : ∀ (n : Nat), CompactSpace (X✝ n)
    inst✝ : ∀ (n : Nat), Nonempty (X✝ n)
    d : Nat → Real := fun n => HPow.hPow (1 / 2) n
    this✝¹ : ∀ (n : Nat), LT.lt 0 (d n)
    u : Nat → GromovHausdorff.GHSpace
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (d …
    X : Nat → Type := fun n => (u n).Rep
    Y : (n : Nat) → GromovHausdorff.AuxGluingStruct (X n) := GromovHausdorff.auxGl …
    E : ∀ (n : Nat), Eq (Metric.GlueSpace ⋯ ⋯) (Y (HAdd.hAdd n 1)).Space
    c : (n : Nat) → Metric.GlueSpace ⋯ ⋯ → (Y (HAdd.hAdd n 1)).Space := fun n => c …
    ic : ∀ (n : Nat), Isometry (c n)
    f : (n : Nat) → (Y n).Space → (Y (HAdd.hAdd n 1)).Space := fun n => Function.c …
    I : ∀ (n : Nat), Isometry (f n)
    Z0 : Type := Metric.InductiveLimit I
    Z : Type := UniformSpace.Completion Z0
    Φ : (n : Nat) → (Y n).Space → Metric.InductiveLimit I := Metric.toInductiveLim …
    coeZ : Z0 → UniformSpace.Completion Z0 := ↑Z0
    X2 : Nat → Set (UniformSpace.Completion Z0) := fun n => Set.range (Function.co …
    isom : ∀ (n : Nat), Isometry (Function.comp coeZ (Function.comp (Φ n) (Y n).em …
    X2n : ∀ (n : Nat), Eq (X2 n) (Set.range (Function.comp (Function.comp coeZ (Fu …
    X2nsucc : ∀ (n : Nat), Eq (X2 n.succ) (Set.range (Function.comp (Function.comp …
    D2 : ∀ (n : Nat), LT.lt (Metric.hausdorffDist (X2 n) (X2 n.succ)) (d n)
    X3 : Nat → TopologicalSpace.NonemptyCompacts Z := fun n => { carrier := X2 n,  …
    this✝ : CauchySeq X3
    L : TopologicalSpace.NonemptyCompacts Z
    hL : Filter.Tendsto X3 Filter.atTop (nhds L)
    this : ∀ (n : Nat), Eq (Function.comp TopologicalSpace.NonemptyCompacts.toGHSp …
    ⊢ Filter.Tendsto TopologicalSpace.NonemptyCompacts.toGHSpace (nhds L) (nhds L. …
  -/
  apply toGHSpace_continuous.tendsto
  /-
    🎉 no goals
  -/


