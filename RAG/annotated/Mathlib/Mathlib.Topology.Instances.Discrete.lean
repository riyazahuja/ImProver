instance (priority := 100) DiscreteTopology.firstCountableTopology [DiscreteTopology α] :
    FirstCountableTopology α where
                                 /-
                                   α : Type u_1
                                   inst✝¹ : TopologicalSpace α
                                   inst✝ : DiscreteTopology α
                                   ⊢ ∀ (a : α), (nhds a).IsCountablyGenerated
                                 -/
  nhds_generated_countable := by rw [nhds_discrete]; exact isCountablyGenerated_pure
                                                     /-
                                                       🎉 no goals
                                                     -/


instance (priority := 100) DiscreteTopology.secondCountableTopology_of_countable
    [hd : DiscreteTopology α] [Countable α] : SecondCountableTopology α :=
  haveI : ∀ i : α, SecondCountableTopology (↥({i} : Set α)) := fun i =>
    { is_open_generated_countable :=
                                           /-
                                             α : Type u_1
                                             inst✝¹ : TopologicalSpace α
                                             hd : DiscreteTopology α
                                             inst✝ : Countable α
                                             i : α
                                             ⊢ Eq instTopologicalSpaceSubtype (TopologicalSpace.generateFrom (Singleton.sin …
                                           -/
        ⟨{univ}, countable_singleton _, by simp only [eq_iff_true_of_subsingleton]⟩ }
                                           /-
                                             🎉 no goals
                                           -/
  secondCountableTopology_of_countable_cover (singletons_open_iff_discrete.mpr hd)
    (iUnion_of_singleton α)


@[deprecated DiscreteTopology.secondCountableTopology_of_countable (since := "2024-03-11")]
theorem DiscreteTopology.secondCountableTopology_of_encodable {α : Type*}
    [TopologicalSpace α] [DiscreteTopology α] [Countable α] : SecondCountableTopology α :=
  DiscreteTopology.secondCountableTopology_of_countable


theorem LinearOrder.bot_topologicalSpace_eq_generateFrom {α} [LinearOrder α] [PredOrder α]
    [SuccOrder α] : (⊥ : TopologicalSpace α) = generateFrom { s | ∃ a, s = Ioi a ∨ s = Iio a } := by
  /-
    α : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    inst✝ : SuccOrder α
    ⊢ Eq Bot.bot (TopologicalSpace.generateFrom (setOf fun s => Exists fun a => Or …
  -/
  let _ := Preorder.topology α
  /-
    α : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    inst✝ : SuccOrder α
    x✝ : TopologicalSpace α := Preorder.topology α
    ⊢ Eq Bot.bot (TopologicalSpace.generateFrom (setOf fun s => Exists fun a => Or …
  -/
  have : OrderTopology α := ⟨rfl⟩
  /-
    α : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    inst✝ : SuccOrder α
    x✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    ⊢ Eq Bot.bot (TopologicalSpace.generateFrom (setOf fun s => Exists fun a => Or …
  -/
  exact DiscreteTopology.of_predOrder_succOrder.eq_bot.symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-02")]
alias bot_topologicalSpace_eq_generateFrom_of_pred_succOrder :=
  LinearOrder.bot_topologicalSpace_eq_generateFrom


theorem discreteTopology_iff_orderTopology_of_pred_succ [LinearOrder α] [PredOrder α]
    [SuccOrder α] : DiscreteTopology α ↔ OrderTopology α := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    inst✝ : SuccOrder α
    ⊢ Iff (DiscreteTopology α) (OrderTopology α)
  -/
  refine ⟨fun h ↦ ⟨?_⟩, fun h ↦ .of_predOrder_succOrder⟩
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    inst✝ : SuccOrder α
    h : DiscreteTopology α
    ⊢ Eq inst✝³ (TopologicalSpace.generateFrom (setOf fun s => Exists fun a => Or  …
  -/
  rw [h.eq_bot, LinearOrder.bot_topologicalSpace_eq_generateFrom]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-02")]
alias discreteTopology_iff_orderTopology_of_pred_succ' :=
  discreteTopology_iff_orderTopology_of_pred_succ


instance OrderTopology.of_discreteTopology [LinearOrder α] [PredOrder α] [SuccOrder α]
    [DiscreteTopology α] : OrderTopology α :=
  discreteTopology_iff_orderTopology_of_pred_succ.mp ‹_›

