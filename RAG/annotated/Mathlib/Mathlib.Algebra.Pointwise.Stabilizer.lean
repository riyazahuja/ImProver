@[to_additive (attr := simp)]
lemma stabilizer_empty : stabilizer G (∅ : Set α) = ⊤ :=
  Subgroup.coe_eq_univ.1 <| eq_univ_of_forall fun _a ↦ smul_set_empty


@[to_additive (attr := simp)]
lemma stabilizer_univ : stabilizer G (Set.univ : Set α) = ⊤ := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    ⊢ Eq (MulAction.stabilizer G Set.univ) Top.top
  -/
  ext
  /-
    case h
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x✝ : G
    ⊢ Iff (Membership.mem (MulAction.stabilizer G Set.univ) x✝) (Membership.mem To …
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
                                                                                       /-
                                                                                         G : Type u_1
                                                                                         α : Type u_3
                                                                                         inst✝¹ : Group G
                                                                                         inst✝ : MulAction G α
                                                                                         b : α
                                                                                         ⊢ Eq (MulAction.stabilizer G (Singleton.singleton b)) (MulAction.stabilizer G b)
                                                                                       -/
lemma stabilizer_singleton (b : α) : stabilizer G ({b} : Set α) = stabilizer G b := by ext; simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[to_additive]
lemma mem_stabilizer_set {s : Set α} : a ∈ stabilizer G s ↔ ∀ b, a • b ∈ s ↔ b ∈ s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    s : Set α
    ⊢ Iff (Membership.mem (MulAction.stabilizer G s) a) (∀ (b : α), Iff (Membershi …
  -/
  refine mem_stabilizer_iff.trans ⟨fun h b ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      G : Type u_1
      α : Type u_3
      inst✝¹ : Group G
      inst✝ : MulAction G α
      a : G
      s : Set α
      h : Eq (HSMul.hSMul a s) s
      b : α
      ⊢ Iff (Membership.mem s (HSMul.hSMul a b)) (Membership.mem s b)
    -/
  · rw [← (smul_mem_smul_set_iff : a • b ∈ _ ↔ _), h]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    s : Set α
    h : ∀ (b : α), Iff (Membership.mem s (HSMul.hSMul a b)) (Membership.mem s b)
    ⊢ Eq (HSMul.hSMul a s) s
  -/
  simp_rw [Set.ext_iff, mem_smul_set_iff_inv_smul_mem]
  /-
    case refine_2
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    s : Set α
    h : ∀ (b : α), Iff (Membership.mem s (HSMul.hSMul a b)) (Membership.mem s b)
    ⊢ ∀ (x : α), Iff (Membership.mem s (HSMul.hSMul (Inv.inv a) x)) (Membership.me …
  -/
  exact ((MulAction.toPerm a).forall_congr' <| by simp [Iff.comm]).1 h
  /-
    🎉 no goals
  -/


@[to_additive]
lemma map_stabilizer_le (f : G →* H) (s : Set G) :
    (stabilizer G s).map f ≤ stabilizer H (f '' s) := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    f : MonoidHom G H
    s : Set G
    ⊢ LE.le (Subgroup.map f (MulAction.stabilizer G s)) (MulAction.stabilizer H (S …
  -/
  rintro a
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    f : MonoidHom G H
    s : Set G
    a : H
    ⊢ Membership.mem (Subgroup.map f (MulAction.stabilizer G s)) a → Membership.me …
  -/
  simp only [Subgroup.mem_map, mem_stabilizer_iff, exists_prop, forall_exists_index, and_imp]
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    f : MonoidHom G H
    s : Set G
    a : H
    ⊢ ∀ (x : G), Eq (HSMul.hSMul x s) s → Eq (f x) a → Eq (HSMul.hSMul a (Set.imag …
  -/
  rintro a ha rfl
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    f : MonoidHom G H
    s : Set G
    a : G
    ha : Eq (HSMul.hSMul a s) s
    ⊢ Eq (HSMul.hSMul (f a) (Set.image (⇑f) s)) (Set.image (⇑f) s)
  -/
  rw [← image_smul_distrib, ha]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma stabilizer_mul_self (s : Set G) : (stabilizer G s : Set G) * s = s := by
  /-
    G : Type u_1
    inst✝ : Group G
    s : Set G
    ⊢ Eq (HMul.hMul (↑(MulAction.stabilizer G s)) s) s
  -/
  ext
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    s : Set G
    x✝ : G
    ⊢ Iff (Membership.mem (HMul.hMul (↑(MulAction.stabilizer G s)) s) x✝) (Members …
  -/
  refine ⟨?_, fun h ↦ ⟨_, (stabilizer G s).one_mem, _, h, one_mul _⟩⟩
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    s : Set G
    x✝ : G
    ⊢ Membership.mem (HMul.hMul (↑(MulAction.stabilizer G s)) s) x✝ → Membership.m …
  -/
  rintro ⟨a, ha, b, hb, rfl⟩
  /-
    case h.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    s : Set G
    a : G
    ha : Membership.mem (↑(MulAction.stabilizer G s)) a
    b : G
    hb : Membership.mem s b
    ⊢ Membership.mem s ((fun x1 x2 => HMul.hMul x1 x2) a b)
  -/
  rw [← mem_stabilizer_iff.1 ha]
  /-
    case h.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    s : Set G
    a : G
    ha : Membership.mem (↑(MulAction.stabilizer G s)) a
    b : G
    hb : Membership.mem s b
    ⊢ Membership.mem (HSMul.hSMul a s) ((fun x1 x2 => HMul.hMul x1 x2) a b)
  -/
  exact smul_mem_smul_set hb
  /-
    🎉 no goals
  -/


@[to_additive]
lemma stabilizer_inf_stabilizer_le_stabilizer_apply₂ {f : Set α → Set α → Set α}
    (hf : ∀ a : G, a • f s t = f (a • s) (a • t)) :
                                                                 /-
                                                                   G : Type u_1
                                                                   α : Type u_3
                                                                   inst✝¹ : Group G
                                                                   inst✝ : MulAction G α
                                                                   s t : Set α
                                                                   f : Set α → Set α → Set α
                                                                   hf : ∀ (a : G), Eq (HSMul.hSMul a (f s t)) (f (HSMul.hSMul a s) (HSMul.hSMul a …
                                                                   ⊢ LE.le (Min.min (MulAction.stabilizer G s) (MulAction.stabilizer G t)) (MulAc …
                                                                 -/
    stabilizer G s ⊓ stabilizer G t ≤ stabilizer G (f s t) := by aesop (add simp [SetLike.le_def])
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
lemma stabilizer_inf_stabilizer_le_stabilizer_union :
    stabilizer G s ⊓ stabilizer G t ≤ stabilizer G (s ∪ t) :=
  stabilizer_inf_stabilizer_le_stabilizer_apply₂ fun _ ↦ smul_set_union


@[to_additive]
lemma stabilizer_inf_stabilizer_le_stabilizer_inter :
    stabilizer G s ⊓ stabilizer G t ≤ stabilizer G (s ∩ t) :=
  stabilizer_inf_stabilizer_le_stabilizer_apply₂ fun _ ↦ smul_set_inter


@[to_additive]
lemma stabilizer_inf_stabilizer_le_stabilizer_sdiff :
    stabilizer G s ⊓ stabilizer G t ≤ stabilizer G (s \ t) :=
  stabilizer_inf_stabilizer_le_stabilizer_apply₂ fun _ ↦ smul_set_sdiff


@[to_additive]
lemma stabilizer_union_eq_left (hdisj : Disjoint s t) (hstab : stabilizer G s ≤ stabilizer G t)
    (hstab_union : stabilizer G (s ∪ t) ≤ stabilizer G t) :
    stabilizer G (s ∪ t) = stabilizer G s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s t : Set α
    hdisj : Disjoint s t
    hstab : LE.le (MulAction.stabilizer G s) (MulAction.stabilizer G t)
    hstab_union : LE.le (MulAction.stabilizer G (Union.union s t)) (MulAction.stab …
    ⊢ Eq (MulAction.stabilizer G (Union.union s t)) (MulAction.stabilizer G s)
  -/
  refine le_antisymm ?_ ?_
  · calc
      stabilizer G (s ∪ t)
        ≤ stabilizer G (s ∪ t) ⊓ stabilizer G t := by simpa
      _ ≤ stabilizer G ((s ∪ t) \ t) := stabilizer_inf_stabilizer_le_stabilizer_sdiff
      _ = stabilizer G s := by rw [union_diff_cancel_right]; simpa [← disjoint_iff_inter_eq_empty]
  · calc
      stabilizer G s
        ≤ stabilizer G s ⊓ stabilizer G t := by simpa
      _ ≤ stabilizer G (s ∪ t) := stabilizer_inf_stabilizer_le_stabilizer_union


@[to_additive]
lemma stabilizer_union_eq_right (hdisj : Disjoint s t) (hstab : stabilizer G t ≤ stabilizer G s)
    (hstab_union : stabilizer G (s ∪ t) ≤ stabilizer G s)  :
    stabilizer G (s ∪ t) = stabilizer G t := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s t : Set α
    hdisj : Disjoint s t
    hstab : LE.le (MulAction.stabilizer G t) (MulAction.stabilizer G s)
    hstab_union : LE.le (MulAction.stabilizer G (Union.union s t)) (MulAction.stab …
    ⊢ Eq (MulAction.stabilizer G (Union.union s t)) (MulAction.stabilizer G t)
  -/
  rw [union_comm, stabilizer_union_eq_left hdisj.symm hstab (union_comm .. ▸ hstab_union)]
  /-
    🎉 no goals
  -/


open scoped RightActions in
@[to_additive]
lemma op_smul_set_stabilizer_subset (ha : a ∈ s) : (stabilizer G s : Set G) <• a ⊆ s :=
                                      /-
                                        G : Type u_1
                                        inst✝ : Group G
                                        a : G
                                        s : Set G
                                        ha : Membership.mem s a
                                        b : G
                                        hb : Membership.mem (↑(MulAction.stabilizer G s)) b
                                        ⊢ Membership.mem s (HSMul.hSMul (MulOpposite.op a) b)
                                      -/
  smul_set_subset_iff.2 fun b hb ↦ by rw [← hb]; exact smul_mem_smul_set ha
                                                 /-
                                                   🎉 no goals
                                                 -/


@[to_additive]
lemma stabilizer_subset_div_right (ha : a ∈ s) : ↑(stabilizer G s) ⊆ s / {a} := fun b hb ↦
         /-
           G : Type u_1
           inst✝ : Group G
           a : G
           s : Set G
           ha : Membership.mem s a
           b : G
           hb : Membership.mem (↑(MulAction.stabilizer G s)) b
           ⊢ Membership.mem s (HMul.hMul b a)
         -/
  ⟨_, by rwa [← smul_eq_mul, mem_stabilizer_set.1 hb], _, mem_singleton _, mul_div_cancel_right _ _⟩
         /-
           🎉 no goals
         -/


@[to_additive]
lemma stabilizer_finite (hs₀ : s.Nonempty) (hs : s.Finite) : (stabilizer G s : Set G).Finite := by
  /-
    G : Type u_1
    inst✝ : Group G
    s : Set G
    hs₀ : s.Nonempty
    hs : s.Finite
    ⊢ (↑(MulAction.stabilizer G s)).Finite
  -/
  obtain ⟨a, ha⟩ := hs₀
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    s : Set G
    hs : s.Finite
    a : G
    ha : Membership.mem s a
    ⊢ (↑(MulAction.stabilizer G s)).Finite
  -/
  exact (hs.div <| finite_singleton _).subset <| stabilizer_subset_div_right ha
  /-
    🎉 no goals
  -/


@[to_additive]
lemma smul_set_stabilizer_subset (ha : a ∈ s) : a • (stabilizer G s : Set G) ⊆ s := by
  /-
    G : Type u_1
    inst✝ : CommGroup G
    s : Set G
    a : G
    ha : Membership.mem s a
    ⊢ HasSubset.Subset (HSMul.hSMul a ↑(MulAction.stabilizer G s)) s
  -/
  simpa using op_smul_set_stabilizer_subset ha
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma stabilizer_subgroup (s : Subgroup G) : stabilizer G (s : Set G) = s := by
  /-
    G : Type u_1
    inst✝ : Group G
    s : Subgroup G
    ⊢ Eq (MulAction.stabilizer G ↑s) s
  -/
  simp_rw [SetLike.ext_iff, mem_stabilizer_set]
  /-
    G : Type u_1
    inst✝ : Group G
    s : Subgroup G
    ⊢ ∀ (x : G), Iff (∀ (b : G), Iff (Membership.mem (↑s) (HSMul.hSMul x b)) (Memb …
  -/
  refine fun a ↦ ⟨fun h ↦ ?_, fun ha b ↦ s.mul_mem_cancel_left ha⟩
  /-
    G : Type u_1
    inst✝ : Group G
    s : Subgroup G
    a : G
    h : ∀ (b : G), Iff (Membership.mem (↑s) (HSMul.hSMul a b)) (Membership.mem (↑s …
    ⊢ Membership.mem s a
  -/
  simpa only [smul_eq_mul, SetLike.mem_coe, mul_one] using (h 1).2 s.one_mem
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma stabilizer_op_subgroup (s : Subgroup G) : stabilizer Gᵐᵒᵖ (s : Set G) = s.op := by
  /-
    G : Type u_1
    inst✝ : Group G
    s : Subgroup G
    ⊢ Eq (MulAction.stabilizer (MulOpposite G) ↑s) s.op
  -/
  simp_rw [SetLike.ext_iff, mem_stabilizer_set]
  /-
    G : Type u_1
    inst✝ : Group G
    s : Subgroup G
    ⊢ ∀ (x : MulOpposite G), Iff (∀ (b : G), Iff (Membership.mem (↑s) (HSMul.hSMul …
  -/
  simp only [smul_eq_mul_unop, SetLike.mem_coe, Subgroup.mem_op, «forall», unop_op]
  /-
    G : Type u_1
    inst✝ : Group G
    s : Subgroup G
    ⊢ ∀ (a : G), Iff (∀ (b : G), Iff (Membership.mem s (HMul.hMul b a)) (Membershi …
  -/
  refine fun a ↦ ⟨fun h ↦ ?_, fun ha b ↦ s.mul_mem_cancel_right ha⟩
  /-
    G : Type u_1
    inst✝ : Group G
    s : Subgroup G
    a : G
    h : ∀ (b : G), Iff (Membership.mem s (HMul.hMul b a)) (Membership.mem s b)
    ⊢ Membership.mem s a
  -/
  simpa only [op_smul_eq_mul, SetLike.mem_coe, one_mul] using (h 1).2 s.one_mem
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma stabilizer_subgroup_op (s : Subgroup Gᵐᵒᵖ) : stabilizer G (s : Set Gᵐᵒᵖ) = s.unop := by
  /-
    G : Type u_1
    inst✝ : Group G
    s : Subgroup (MulOpposite G)
    ⊢ Eq (MulAction.stabilizer G ↑s) s.unop
  -/
  simp_rw [SetLike.ext_iff, mem_stabilizer_set]
  /-
    G : Type u_1
    inst✝ : Group G
    s : Subgroup (MulOpposite G)
    ⊢ ∀ (x : G), Iff (∀ (b : MulOpposite G), Iff (Membership.mem (↑s) (HSMul.hSMul …
  -/
  refine fun a ↦ ⟨fun h ↦ ?_, fun ha b ↦ s.mul_mem_cancel_right ha⟩
  /-
    G : Type u_1
    inst✝ : Group G
    s : Subgroup (MulOpposite G)
    a : G
    h : ∀ (b : MulOpposite G), Iff (Membership.mem (↑s) (HSMul.hSMul a b)) (Member …
    ⊢ Membership.mem s.unop a
  -/
  have : 1 * MulOpposite.op a ∈ s := (h 1).2 s.one_mem
  /-
    G : Type u_1
    inst✝ : Group G
    s : Subgroup (MulOpposite G)
    a : G
    h : ∀ (b : MulOpposite G), Iff (Membership.mem (↑s) (HSMul.hSMul a b)) (Member …
    this : Membership.mem s (HMul.hMul 1 (MulOpposite.op a))
    ⊢ Membership.mem s.unop a
  -/
  simpa only [op_smul_eq_mul, SetLike.mem_coe, one_mul] using this
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp, norm_cast)]
lemma stabilizer_coe_finset (s : Finset α) : stabilizer G (s : Set α) = stabilizer G s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝² : Group G
    inst✝¹ : MulAction G α
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (MulAction.stabilizer G ↑s) (MulAction.stabilizer G s)
  -/
  ext; simp [← Finset.coe_inj]
       /-
         🎉 no goals
       -/


@[to_additive (attr := simp)]
lemma stabilizer_finset_empty : stabilizer G (∅ : Finset α) = ⊤ :=
  Subgroup.coe_eq_univ.1 <| eq_univ_of_forall Finset.smul_finset_empty


@[to_additive (attr := simp)]
lemma stabilizer_finset_univ [Fintype α] : stabilizer G (Finset.univ : Finset α) = ⊤ := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝³ : Group G
    inst✝² : MulAction G α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Eq (MulAction.stabilizer G Finset.univ) Top.top
  -/
  ext
  /-
    case h
    G : Type u_1
    α : Type u_3
    inst✝³ : Group G
    inst✝² : MulAction G α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x✝ : G
    ⊢ Iff (Membership.mem (MulAction.stabilizer G Finset.univ) x✝) (Membership.mem …
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma stabilizer_finset_singleton (b : α) : stabilizer G ({b} : Finset α) = stabilizer G b := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝² : Group G
    inst✝¹ : MulAction G α
    inst✝ : DecidableEq α
    b : α
    ⊢ Eq (MulAction.stabilizer G (Singleton.singleton b)) (MulAction.stabilizer G b)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[to_additive]
lemma mem_stabilizer_finset {s : Finset α} : a ∈ stabilizer G s ↔ ∀ b, a • b ∈ s ↔ b ∈ s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝² : Group G
    inst✝¹ : MulAction G α
    a : G
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Iff (Membership.mem (MulAction.stabilizer G s) a) (∀ (b : α), Iff (Membershi …
  -/
  simp_rw [← stabilizer_coe_finset, mem_stabilizer_set, Finset.mem_coe]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mem_stabilizer_finset_iff_subset_smul_finset {s : Finset α} :
    a ∈ stabilizer G s ↔ s ⊆ a • s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝² : Group G
    inst✝¹ : MulAction G α
    a : G
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Iff (Membership.mem (MulAction.stabilizer G s) a) (HasSubset.Subset s (HSMul …
  -/
  rw [mem_stabilizer_iff, Finset.subset_iff_eq_of_card_le (Finset.card_smul_finset _ _).le, eq_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mem_stabilizer_finset_iff_smul_finset_subset {s : Finset α} :
    a ∈ stabilizer G s ↔ a • s ⊆ s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝² : Group G
    inst✝¹ : MulAction G α
    a : G
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Iff (Membership.mem (MulAction.stabilizer G s) a) (HasSubset.Subset (HSMul.h …
  -/
  rw [mem_stabilizer_iff, Finset.subset_iff_eq_of_card_le (Finset.card_smul_finset _ _).ge]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mem_stabilizer_finset' {s : Finset α} : a ∈ stabilizer G s ↔ ∀ ⦃b⦄, b ∈ s → a • b ∈ s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝² : Group G
    inst✝¹ : MulAction G α
    a : G
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Iff (Membership.mem (MulAction.stabilizer G s) a) (∀ ⦃b : α⦄, Membership.mem …
  -/
  rw [← Subgroup.inv_mem_iff, mem_stabilizer_finset_iff_subset_smul_finset]
  /-
    G : Type u_1
    α : Type u_3
    inst✝² : Group G
    inst✝¹ : MulAction G α
    a : G
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Iff (HasSubset.Subset s (HSMul.hSMul (Inv.inv a) s)) (∀ ⦃b : α⦄, Membership. …
  -/
  simp_rw [← Finset.mem_inv_smul_finset_iff, Finset.subset_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mem_stabilizer_set_iff_subset_smul_set {s : Set α} (hs : s.Finite) :
    a ∈ stabilizer G s ↔ s ⊆ a • s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    s : Set α
    hs : s.Finite
    ⊢ Iff (Membership.mem (MulAction.stabilizer G s) a) (HasSubset.Subset s (HSMul …
  -/
  lift s to Finset α using hs
  classical
  rw [stabilizer_coe_finset, mem_stabilizer_finset_iff_subset_smul_finset, ← Finset.coe_smul_finset,
    Finset.coe_subset]


@[to_additive]
lemma mem_stabilizer_set_iff_smul_set_subset {s : Set α} (hs : s.Finite) :
    a ∈ stabilizer G s ↔ a • s ⊆ s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    s : Set α
    hs : s.Finite
    ⊢ Iff (Membership.mem (MulAction.stabilizer G s) a) (HasSubset.Subset (HSMul.h …
  -/
  lift s to Finset α using hs
  classical
  rw [stabilizer_coe_finset, mem_stabilizer_finset_iff_smul_finset_subset, ← Finset.coe_smul_finset,
    Finset.coe_subset]


@[deprecated (since := "2024-11-25")]
alias mem_stabilizer_of_finite_iff_smul_le := mem_stabilizer_set_iff_subset_smul_set


@[deprecated (since := "2024-11-25")]
alias mem_stabilizer_of_finite_iff_le_smul := mem_stabilizer_set_iff_smul_set_subset


@[to_additive]
lemma mem_stabilizer_set' {s : Set α} (hs : s.Finite) :
    a ∈ stabilizer G s ↔ ∀ ⦃b⦄, b ∈ s → a • b ∈ s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    s : Set α
    hs : s.Finite
    ⊢ Iff (Membership.mem (MulAction.stabilizer G s) a) (∀ ⦃b : α⦄, Membership.mem …
  -/
  lift s to Finset α using hs
  /-
    case intro
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    s : Finset α
    ⊢ Iff (Membership.mem (MulAction.stabilizer G ↑s) a) (∀ ⦃b : α⦄, Membership.me …
  -/
  classical simp [-mem_stabilizer_iff, mem_stabilizer_finset']
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
                                                         /-
                                                           G : Type u_1
                                                           inst✝ : CommGroup G
                                                           s : Set G
                                                           ⊢ Eq (HMul.hMul s ↑(MulAction.stabilizer G s)) s
                                                         -/
lemma mul_stabilizer_self : s * stabilizer G s = s := by rw [mul_comm, stabilizer_mul_self]
                                                         /-
                                                           🎉 no goals
                                                         -/


local notation " Q " => G ⧸ stabilizer G s

local notation " q " => ((↑) : G → Q)


@[to_additive]
lemma stabilizer_image_coe_quotient : stabilizer Q (q '' s) = ⊥ := by
  /-
    G : Type u_1
    inst✝ : CommGroup G
    s : Set G
    ⊢ Eq (MulAction.stabilizer (HasQuotient.Quotient G (MulAction.stabilizer G s)) …
  -/
  ext a
  /-
    case h
    G : Type u_1
    inst✝ : CommGroup G
    s : Set G
    a : HasQuotient.Quotient G (MulAction.stabilizer G s)
    ⊢ Iff (Membership.mem (MulAction.stabilizer (HasQuotient.Quotient G (MulAction …
  -/
  induction' a using QuotientGroup.induction_on with a
  /-
    case h.H
    G : Type u_1
    inst✝ : CommGroup G
    s : Set G
    a : G
    ⊢ Iff (Membership.mem (MulAction.stabilizer (HasQuotient.Quotient G (MulAction …
  -/
  simp only [mem_stabilizer_iff, Subgroup.mem_bot, QuotientGroup.eq_one_iff]
  have : q a • q '' s = q '' (a • s) :=
    (image_smul_distrib (QuotientGroup.mk' <| stabilizer G s) _ _).symm
  /-
    case h.H
    G : Type u_1
    inst✝ : CommGroup G
    s : Set G
    a : G
    this : Eq (HSMul.hSMul (↑a) (Set.image QuotientGroup.mk s)) (Set.image Quotien …
    ⊢ Iff (Eq (HSMul.hSMul (↑a) (Set.image QuotientGroup.mk s)) (Set.image Quotien …
  -/
  rw [this]
  /-
    case h.H
    G : Type u_1
    inst✝ : CommGroup G
    s : Set G
    a : G
    this : Eq (HSMul.hSMul (↑a) (Set.image QuotientGroup.mk s)) (Set.image Quotien …
    ⊢ Iff (Eq (Set.image QuotientGroup.mk (HSMul.hSMul a s)) (Set.image QuotientGr …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ by rw [h]⟩
  /-
    case h.H
    G : Type u_1
    inst✝ : CommGroup G
    s : Set G
    a : G
    this : Eq (HSMul.hSMul (↑a) (Set.image QuotientGroup.mk s)) (Set.image Quotien …
    h : Eq (Set.image QuotientGroup.mk (HSMul.hSMul a s)) (Set.image QuotientGroup …
    ⊢ Eq (HSMul.hSMul a s) s
  -/
  rwa [QuotientGroup.image_coe_inj, mul_smul_comm, stabilizer_mul_self] at h
  /-
    🎉 no goals
  -/


