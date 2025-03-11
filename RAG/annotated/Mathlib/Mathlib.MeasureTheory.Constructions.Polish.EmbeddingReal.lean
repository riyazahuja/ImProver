theorem exists_nat_measurableEquiv_range_coe_fin_of_finite [Finite α] :
    ∃ n : ℕ, Nonempty (α ≃ᵐ range ((↑) : Fin n → ℝ)) := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : StandardBorelSpace α
    inst✝ : Finite α
    ⊢ Exists fun n => Nonempty (MeasurableEquiv α ↑(Set.range fun x => ↑↑x))
  -/
  obtain ⟨n, ⟨n_equiv⟩⟩ := Finite.exists_equiv_fin α
  /-
    case intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : StandardBorelSpace α
    inst✝ : Finite α
    n : Nat
    n_equiv : Equiv α (Fin n)
    ⊢ Exists fun n => Nonempty (MeasurableEquiv α ↑(Set.range fun x => ↑↑x))
  -/
  refine ⟨n, ⟨PolishSpace.Equiv.measurableEquiv (n_equiv.trans ?_)⟩⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : StandardBorelSpace α
    inst✝ : Finite α
    n : Nat
    n_equiv : Equiv α (Fin n)
    ⊢ Equiv (Fin n) ↑(Set.range fun x => ↑↑x)
  -/
  exact Equiv.ofInjective _ (Nat.cast_injective.comp Fin.val_injective)
  /-
    🎉 no goals
  -/


theorem measurableEquiv_range_coe_nat_of_infinite_of_countable [Infinite α] [Countable α] :
    Nonempty (α ≃ᵐ range ((↑) : ℕ → ℝ)) := by
  have : PolishSpace (range ((↑) : ℕ → ℝ)) :=
    Nat.isClosedEmbedding_coe_real.isClosedMap.isClosed_range.polishSpace
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    inst✝² : StandardBorelSpace α
    inst✝¹ : Infinite α
    inst✝ : Countable α
    this : PolishSpace ↑(Set.range Nat.cast)
    ⊢ Nonempty (MeasurableEquiv α ↑(Set.range Nat.cast))
  -/
  refine ⟨PolishSpace.Equiv.measurableEquiv ?_⟩
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    inst✝² : StandardBorelSpace α
    inst✝¹ : Infinite α
    inst✝ : Countable α
    this : PolishSpace ↑(Set.range Nat.cast)
    ⊢ Equiv α ↑(Set.range Nat.cast)
  -/
  refine (nonempty_equiv_of_countable.some : α ≃ ℕ).trans ?_
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    inst✝² : StandardBorelSpace α
    inst✝¹ : Infinite α
    inst✝ : Countable α
    this : PolishSpace ↑(Set.range Nat.cast)
    ⊢ Equiv Nat ↑(Set.range Nat.cast)
  -/
  exact Equiv.ofInjective ((↑) : ℕ → ℝ) Nat.cast_injective
  /-
    🎉 no goals
  -/


/-- Any standard Borel space is measurably equivalent to a subset of the reals. -/
theorem exists_subset_real_measurableEquiv : ∃ s : Set ℝ, MeasurableSet s ∧ Nonempty (α ≃ᵐ s) := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    ⊢ Exists fun s => And (MeasurableSet s) (Nonempty (MeasurableEquiv α ↑s))
  -/
  by_cases hα : Countable α
    /-
      case pos
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      inst✝ : StandardBorelSpace α
      hα : Countable α
      ⊢ Exists fun s => And (MeasurableSet s) (Nonempty (MeasurableEquiv α ↑s))
    -/
  · cases finite_or_infinite α
      /-
        case pos.inl
        α : Type u_1
        inst✝¹ : MeasurableSpace α
        inst✝ : StandardBorelSpace α
        hα : Countable α
        h✝ : Finite α
        ⊢ Exists fun s => And (MeasurableSet s) (Nonempty (MeasurableEquiv α ↑s))
      -/
    · obtain ⟨n, h_nonempty_equiv⟩ := exists_nat_measurableEquiv_range_coe_fin_of_finite α
      /-
        case pos.inl.intro
        α : Type u_1
        inst✝¹ : MeasurableSpace α
        inst✝ : StandardBorelSpace α
        hα : Countable α
        h✝ : Finite α
        n : Nat
        h_nonempty_equiv : Nonempty (MeasurableEquiv α ↑(Set.range fun x => ↑↑x))
        ⊢ Exists fun s => And (MeasurableSet s) (Nonempty (MeasurableEquiv α ↑s))
      -/
      refine ⟨_, ?_, h_nonempty_equiv⟩
      /-
        case pos.inl.intro
        α : Type u_1
        inst✝¹ : MeasurableSpace α
        inst✝ : StandardBorelSpace α
        hα : Countable α
        h✝ : Finite α
        n : Nat
        h_nonempty_equiv : Nonempty (MeasurableEquiv α ↑(Set.range fun x => ↑↑x))
        ⊢ MeasurableSet (Set.range fun x => ↑↑x)
      -/
      letI : MeasurableSpace (Fin n) := borel (Fin n)
      /-
        case pos.inl.intro
        α : Type u_1
        inst✝¹ : MeasurableSpace α
        inst✝ : StandardBorelSpace α
        hα : Countable α
        h✝ : Finite α
        n : Nat
        h_nonempty_equiv : Nonempty (MeasurableEquiv α ↑(Set.range fun x => ↑↑x))
        this : MeasurableSpace (Fin n) := borel (Fin n)
        ⊢ MeasurableSet (Set.range fun x => ↑↑x)
      -/
      haveI : BorelSpace (Fin n) := ⟨rfl⟩
      /-
        case pos.inl.intro
        α : Type u_1
        inst✝¹ : MeasurableSpace α
        inst✝ : StandardBorelSpace α
        hα : Countable α
        h✝ : Finite α
        n : Nat
        h_nonempty_equiv : Nonempty (MeasurableEquiv α ↑(Set.range fun x => ↑↑x))
        this✝ : MeasurableSpace (Fin n) := borel (Fin n)
        this : BorelSpace (Fin n)
        ⊢ MeasurableSet (Set.range fun x => ↑↑x)
      -/
      apply MeasurableEmbedding.measurableSet_range (mα := by infer_instance)
      exact continuous_of_discreteTopology.measurableEmbedding
        (Nat.cast_injective.comp Fin.val_injective)
      /-
        case pos.inr
        α : Type u_1
        inst✝¹ : MeasurableSpace α
        inst✝ : StandardBorelSpace α
        hα : Countable α
        h✝ : Infinite α
        ⊢ Exists fun s => And (MeasurableSet s) (Nonempty (MeasurableEquiv α ↑s))
      -/
    · refine ⟨_, ?_, measurableEquiv_range_coe_nat_of_infinite_of_countable α⟩
      /-
        case pos.inr
        α : Type u_1
        inst✝¹ : MeasurableSpace α
        inst✝ : StandardBorelSpace α
        hα : Countable α
        h✝ : Infinite α
        ⊢ MeasurableSet (Set.range Nat.cast)
      -/
      apply MeasurableEmbedding.measurableSet_range (mα := by infer_instance)
      /-
        case pos.inr
        α : Type u_1
        inst✝¹ : MeasurableSpace α
        inst✝ : StandardBorelSpace α
        hα : Countable α
        h✝ : Infinite α
        ⊢ MeasurableEmbedding Nat.cast
      -/
      exact continuous_of_discreteTopology.measurableEmbedding Nat.cast_injective
      /-
        🎉 no goals
      -/
  · refine
      ⟨univ, MeasurableSet.univ,
        ⟨(PolishSpace.measurableEquivOfNotCountable hα ?_ : α ≃ᵐ (univ : Set ℝ))⟩⟩
    /-
      case neg
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      inst✝ : StandardBorelSpace α
      hα : Not (Countable α)
      ⊢ Not (Countable ↑Set.univ)
    -/
    rw [countable_coe_iff]
    /-
      case neg
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      inst✝ : StandardBorelSpace α
      hα : Not (Countable α)
      ⊢ Not Set.univ.Countable
    -/
    exact Cardinal.not_countable_real
    /-
      🎉 no goals
    -/


/-- Any standard Borel space embeds measurably into the reals. -/
theorem exists_measurableEmbedding_real : ∃ f : α → ℝ, MeasurableEmbedding f := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    ⊢ Exists fun f => MeasurableEmbedding f
  -/
  obtain ⟨s, hs, ⟨e⟩⟩ := exists_subset_real_measurableEquiv α
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    s : Set Real
    hs : MeasurableSet s
    e : MeasurableEquiv α ↑s
    ⊢ Exists fun f => MeasurableEmbedding f
  -/
  exact ⟨(↑) ∘ e, (MeasurableEmbedding.subtype_coe hs).comp e.measurableEmbedding⟩
  /-
    🎉 no goals
  -/


/-- A measurable embedding of a standard Borel space into `ℝ`. -/
noncomputable
def embeddingReal (Ω : Type*) [MeasurableSpace Ω] [StandardBorelSpace Ω] : Ω → ℝ :=
  (exists_measurableEmbedding_real Ω).choose


lemma measurableEmbedding_embeddingReal (Ω : Type*) [MeasurableSpace Ω] [StandardBorelSpace Ω] :
    MeasurableEmbedding (embeddingReal Ω) :=
  (exists_measurableEmbedding_real Ω).choose_spec


@[fun_prop]
lemma measurable_embeddingReal (Ω : Type*) [MeasurableSpace Ω] [StandardBorelSpace Ω] :
    Measurable (embeddingReal Ω) :=
  (measurableEmbedding_embeddingReal Ω).measurable


