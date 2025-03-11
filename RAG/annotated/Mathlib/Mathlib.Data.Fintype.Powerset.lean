instance Finset.fintype [Fintype α] : Fintype (Finset α) :=
  ⟨univ.powerset, fun _ => Finset.mem_powerset.2 (Finset.subset_univ _)⟩


@[simp]
theorem Fintype.card_finset [Fintype α] : Fintype.card (Finset α) = 2 ^ Fintype.card α :=
  Finset.card_powerset Finset.univ


@[simp] lemma powerset_univ : (univ : Finset α).powerset = univ :=
                      /-
                        α : Type u_1
                        inst✝ : Fintype α
                        ⊢ Eq ↑Finset.univ.powerset ↑Finset.univ
                      -/
  coe_injective <| by simp [-coe_eq_univ]
                      /-
                        🎉 no goals
                      -/


lemma filter_subset_univ [DecidableEq α] (s : Finset α) :
                                                /-
                                                  α : Type u_1
                                                  inst✝¹ : Fintype α
                                                  inst✝ : DecidableEq α
                                                  s : Finset α
                                                  ⊢ Eq (Finset.filter (fun t => HasSubset.Subset t s) Finset.univ) s.powerset
                                                -/
    ({t | t ⊆ s} : Finset _) = powerset s := by ext; simp
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma powerset_eq_univ : s.powerset = univ ↔ s = univ := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    s : Finset α
    ⊢ Iff (Eq s.powerset Finset.univ) (Eq s Finset.univ)
  -/
  rw [← Finset.powerset_univ, powerset_inj]
  /-
    🎉 no goals
  -/


lemma mem_powersetCard_univ : s ∈ powersetCard k (univ : Finset α) ↔ #s = k :=
  mem_powersetCard.trans <| and_iff_right <| subset_univ _


@[simp] lemma univ_filter_card_eq (k : ℕ) :
                                                                  /-
                                                                    α : Type u_1
                                                                    inst✝ : Fintype α
                                                                    k : Nat
                                                                    ⊢ Eq (Finset.filter (fun s => Eq s.card k) Finset.univ) (Finset.powersetCard k …
                                                                  -/
   ({s | #s = k} : Finset (Finset α)) = univ.powersetCard k := by ext; simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem Fintype.card_finset_len [Fintype α] (k : ℕ) :
    Fintype.card { s : Finset α // #s = k } = Nat.choose (Fintype.card α) k := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    k : Nat
    ⊢ Eq (Fintype.card (Subtype fun s => Eq s.card k)) ((Fintype.card α).choose k)
  -/
  simp [Fintype.subtype_card, Finset.card_univ]
  /-
    🎉 no goals
  -/


instance Set.fintype [Fintype α] : Fintype (Set α) :=
  ⟨(@Finset.univ (Finset α) _).map coeEmb.1, fun s => by
    classical
    refine mem_map.2 ⟨({a | a ∈ s} : Finset _), Finset.mem_univ _, (coe_filter _ _).trans ?_⟩
    simp⟩

-- Not to be confused with `Set.Finite`, the predicate

instance Set.finite' [Finite α] : Finite (Set α) := by
  /-
    α : Type u_1
    inst✝ : Finite α
    ⊢ Finite (Set α)
  -/
  cases nonempty_fintype α
  /-
    case intro
    α : Type u_1
    inst✝ : Finite α
    val✝ : Fintype α
    ⊢ Finite (Set α)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem Fintype.card_set [Fintype α] : Fintype.card (Set α) = 2 ^ Fintype.card α :=
  (Finset.card_map _).trans (Finset.card_powerset _)

