/-- A set `A` is balanced if `a • A` is contained in `A` whenever `a` has norm at most `1`. -/
def Balanced (A : Set E) :=
  ∀ a : 𝕜, ‖a‖ ≤ 1 → a • A ⊆ A


lemma absorbs_iff_norm : Absorbs 𝕜 A B ↔ ∃ r, ∀ c : 𝕜, r ≤ ‖c‖ → B ⊆ c • A :=
                                                                  /-
                                                                    𝕜 : Type u_1
                                                                    E : Type u_3
                                                                    inst✝¹ : SeminormedRing 𝕜
                                                                    inst✝ : SMul 𝕜 E
                                                                    A B : Set E
                                                                    ⊢ Iff (Exists fun i => And True (∀ ⦃x : 𝕜⦄, Membership.mem (Set.preimage Norm. …
                                                                  -/
  Filter.atTop_basis.cobounded_of_norm.eventually_iff.trans <| by simp only [true_and]; rfl
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


alias ⟨_, Absorbs.of_norm⟩ := absorbs_iff_norm


lemma Absorbs.exists_pos (h : Absorbs 𝕜 A B) : ∃ r > 0, ∀ c : 𝕜, r ≤ ‖c‖ → B ⊆ c • A :=
  let ⟨r, hr₁, hr⟩ := (Filter.atTop_basis' 1).cobounded_of_norm.eventually_iff.1 h
  ⟨r, one_pos.trans_le hr₁, hr⟩


theorem balanced_iff_smul_mem : Balanced 𝕜 s ↔ ∀ ⦃a : 𝕜⦄, ‖a‖ ≤ 1 → ∀ ⦃x : E⦄, x ∈ s → a • x ∈ s :=
  forall₂_congr fun _a _ha => smul_set_subset_iff


alias ⟨Balanced.smul_mem, _⟩ := balanced_iff_smul_mem


theorem balanced_iff_closedBall_smul : Balanced 𝕜 s ↔ Metric.closedBall (0 : 𝕜) 1 • s ⊆ s := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝¹ : SeminormedRing 𝕜
    inst✝ : SMul 𝕜 E
    s : Set E
    ⊢ Iff (Balanced 𝕜 s) (HasSubset.Subset (HSMul.hSMul (Metric.closedBall 0 1) s) …
  -/
  simp [balanced_iff_smul_mem, smul_subset_iff]
  /-
    🎉 no goals
  -/


@[simp]
                                                                 /-
                                                                   𝕜 : Type u_1
                                                                   E : Type u_3
                                                                   inst✝¹ : SeminormedRing 𝕜
                                                                   inst✝ : SMul 𝕜 E
                                                                   x✝¹ : 𝕜
                                                                   x✝ : LE.le (Norm.norm x✝¹) 1
                                                                   ⊢ HasSubset.Subset (HSMul.hSMul x✝¹ EmptyCollection.emptyCollection) EmptyColl …
                                                                 -/
theorem balanced_empty : Balanced 𝕜 (∅ : Set E) := fun _ _ => by rw [smul_set_empty]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem balanced_univ : Balanced 𝕜 (univ : Set E) := fun _a _ha => subset_univ _


theorem Balanced.union (hA : Balanced 𝕜 A) (hB : Balanced 𝕜 B) : Balanced 𝕜 (A ∪ B) := fun _a ha =>
  smul_set_union.subset.trans <| union_subset_union (hA _ ha) <| hB _ ha


theorem Balanced.inter (hA : Balanced 𝕜 A) (hB : Balanced 𝕜 B) : Balanced 𝕜 (A ∩ B) := fun _a ha =>
  smul_set_inter_subset.trans <| inter_subset_inter (hA _ ha) <| hB _ ha


theorem balanced_iUnion {f : ι → Set E} (h : ∀ i, Balanced 𝕜 (f i)) : Balanced 𝕜 (⋃ i, f i) :=
  fun _a ha => (smul_set_iUnion _ _).subset.trans <| iUnion_mono fun _ => h _ _ ha


theorem balanced_iUnion₂ {f : ∀ i, κ i → Set E} (h : ∀ i j, Balanced 𝕜 (f i j)) :
    Balanced 𝕜 (⋃ (i) (j), f i j) :=
  balanced_iUnion fun _ => balanced_iUnion <| h _


theorem Balanced.sInter {S : Set (Set E)} (h : ∀ s ∈ S, Balanced 𝕜 s) : Balanced 𝕜 (⋂₀ S) :=
                                                              /-
                                                                𝕜 : Type u_1
                                                                E : Type u_3
                                                                inst✝¹ : SeminormedRing 𝕜
                                                                inst✝ : SMul 𝕜 E
                                                                S : Set (Set E)
                                                                h : ∀ (s : Set E), Membership.mem S s → Balanced 𝕜 s
                                                                x✝³ : 𝕜
                                                                x✝² : LE.le (Norm.norm x✝³) 1
                                                                x✝¹ : E
                                                                x✝ : Membership.mem (Set.iInter fun s => Set.iInter fun h => HSMul.hSMul x✝³ s …
                                                                ⊢ Membership.mem S.sInter x✝¹
                                                              -/
  fun _ _ => (smul_set_sInter_subset ..).trans (fun _ _ => by aesop)
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem balanced_iInter {f : ι → Set E} (h : ∀ i, Balanced 𝕜 (f i)) : Balanced 𝕜 (⋂ i, f i) :=
  fun _a ha => (smul_set_iInter_subset _ _).trans <| iInter_mono fun _ => h _ _ ha


theorem balanced_iInter₂ {f : ∀ i, κ i → Set E} (h : ∀ i j, Balanced 𝕜 (f i j)) :
    Balanced 𝕜 (⋂ (i) (j), f i j) :=
  balanced_iInter fun _ => balanced_iInter <| h _


theorem Balanced.smul (a : 𝕝) (hs : Balanced 𝕜 s) : Balanced 𝕜 (a • s) := fun _b hb =>
  (smul_comm _ _ _).subset.trans <| smul_set_mono <| hs _ hb


theorem Balanced.neg : Balanced 𝕜 s → Balanced 𝕜 (-s) :=
  forall₂_imp fun _ _ h => (smul_set_neg _ _).subset.trans <| neg_subset_neg.2 h


@[simp]
theorem balanced_neg : Balanced 𝕜 (-s) ↔ Balanced 𝕜 s :=
  ⟨fun h ↦ neg_neg s ▸ h.neg, fun h ↦ h.neg⟩


theorem Balanced.neg_mem_iff [NormOneClass 𝕜] (h : Balanced 𝕜 s) {x : E} : -x ∈ s ↔ x ∈ s :=
               /-
                 𝕜 : Type u_1
                 E : Type u_3
                 inst✝³ : SeminormedRing 𝕜
                 inst✝² : AddCommGroup E
                 inst✝¹ : Module 𝕜 E
                 s : Set E
                 inst✝ : NormOneClass 𝕜
                 h : Balanced 𝕜 s
                 x : E
                 hx : Membership.mem s (Neg.neg x)
                 ⊢ Membership.mem s x
               -/
  ⟨fun hx ↦ by simpa using h.smul_mem (a := -1) (by simp) hx,
               /-
                 🎉 no goals
               -/
                /-
                  𝕜 : Type u_1
                  E : Type u_3
                  inst✝³ : SeminormedRing 𝕜
                  inst✝² : AddCommGroup E
                  inst✝¹ : Module 𝕜 E
                  s : Set E
                  inst✝ : NormOneClass 𝕜
                  h : Balanced 𝕜 s
                  x : E
                  hx : Membership.mem s x
                  ⊢ Membership.mem s (Neg.neg x)
                -/
    fun hx ↦ by simpa using h.smul_mem (a := -1) (by simp) hx⟩
                /-
                  🎉 no goals
                -/


theorem Balanced.neg_eq [NormOneClass 𝕜] (h : Balanced 𝕜 s) : -s = s :=
  Set.ext fun _ ↦ h.neg_mem_iff


theorem Balanced.add (hs : Balanced 𝕜 s) (ht : Balanced 𝕜 t) : Balanced 𝕜 (s + t) := fun _a ha =>
  (smul_add _ _ _).subset.trans <| add_subset_add (hs _ ha) <| ht _ ha


theorem Balanced.sub (hs : Balanced 𝕜 s) (ht : Balanced 𝕜 t) : Balanced 𝕜 (s - t) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Balanced 𝕜 s
    ht : Balanced 𝕜 t
    ⊢ Balanced 𝕜 (HSub.hSub s t)
  -/
  simp_rw [sub_eq_add_neg]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Balanced 𝕜 s
    ht : Balanced 𝕜 t
    ⊢ Balanced 𝕜 (HAdd.hAdd s (Neg.neg t))
  -/
  exact hs.add ht.neg
  /-
    🎉 no goals
  -/


theorem balanced_zero : Balanced 𝕜 (0 : Set E) := fun _a _ha => (smul_zero _).subset


theorem absorbs_iff_eventually_nhdsWithin_zero :
    Absorbs 𝕜 s t ↔ ∀ᶠ c : 𝕜 in 𝓝[≠] 0, MapsTo (c • ·) t s := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    ⊢ Iff (Absorbs 𝕜 s t) (Filter.Eventually (fun c => Set.MapsTo (fun x => HSMul. …
  -/
  rw [absorbs_iff_eventually_cobounded_mapsTo, ← Filter.inv_cobounded₀]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


alias ⟨Absorbs.eventually_nhdsWithin_zero, _⟩ := absorbs_iff_eventually_nhdsWithin_zero


theorem absorbent_iff_eventually_nhdsWithin_zero :
    Absorbent 𝕜 s ↔ ∀ x : E, ∀ᶠ c : 𝕜 in 𝓝[≠] 0, c • x ∈ s :=
                           /-
                             𝕜 : Type u_1
                             E : Type u_3
                             inst✝² : NormedDivisionRing 𝕜
                             inst✝¹ : AddCommGroup E
                             inst✝ : Module 𝕜 E
                             s : Set E
                             x : E
                             ⊢ Iff (Absorbs 𝕜 s (Singleton.singleton x)) (Filter.Eventually (fun c => Membe …
                           -/
  forall_congr' fun x ↦ by simp only [absorbs_iff_eventually_nhdsWithin_zero, mapsTo_singleton]
                           /-
                             🎉 no goals
                           -/


alias ⟨Absorbent.eventually_nhdsWithin_zero, _⟩ := absorbent_iff_eventually_nhdsWithin_zero


theorem absorbs_iff_eventually_nhds_zero (h₀ : 0 ∈ s) :
    Absorbs 𝕜 s t ↔ ∀ᶠ c : 𝕜 in 𝓝 0, MapsTo (c • ·) t s := by
  rw [← nhdsWithin_compl_singleton_sup_pure, Filter.eventually_sup, Filter.eventually_pure,
    ← absorbs_iff_eventually_nhdsWithin_zero, and_iff_left]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    h₀ : Membership.mem s 0
    ⊢ Set.MapsTo (fun x => HSMul.hSMul 0 x) t s
  -/
  intro x _
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NormedDivisionRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    h₀ : Membership.mem s 0
    x : E
    a✝ : Membership.mem t x
    ⊢ Membership.mem s ((fun x => HSMul.hSMul 0 x) x)
  -/
  simpa only [zero_smul]
  /-
    🎉 no goals
  -/


theorem Absorbs.eventually_nhds_zero (h : Absorbs 𝕜 s t) (h₀ : 0 ∈ s) :
    ∀ᶠ c : 𝕜 in 𝓝 0, MapsTo (c • ·) t s :=
  (absorbs_iff_eventually_nhds_zero h₀).1 h


/-- Scalar multiplication (by possibly different types) of a balanced set is monotone. -/
theorem Balanced.smul_mono (hs : Balanced 𝕝 s) {a : 𝕝} {b : 𝕜} (h : ‖a‖ ≤ ‖b‖) : a • s ⊆ b • s := by
  /-
    𝕜 : Type u_1
    𝕝 : Type u_2
    E : Type u_3
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : NormedRing 𝕝
    inst✝⁴ : NormedSpace 𝕜 𝕝
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : SMulWithZero 𝕝 E
    inst✝ : IsScalarTower 𝕜 𝕝 E
    s : Set E
    hs : Balanced 𝕝 s
    a : 𝕝
    b : 𝕜
    h : LE.le (Norm.norm a) (Norm.norm b)
    ⊢ HasSubset.Subset (HSMul.hSMul a s) (HSMul.hSMul b s)
  -/
  obtain rfl | hb := eq_or_ne b 0
    /-
      case inl
      𝕜 : Type u_1
      𝕝 : Type u_2
      E : Type u_3
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : NormedRing 𝕝
      inst✝⁴ : NormedSpace 𝕜 𝕝
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : SMulWithZero 𝕝 E
      inst✝ : IsScalarTower 𝕜 𝕝 E
      s : Set E
      hs : Balanced 𝕝 s
      a : 𝕝
      h : LE.le (Norm.norm a) (Norm.norm 0)
      ⊢ HasSubset.Subset (HSMul.hSMul a s) (HSMul.hSMul 0 s)
    -/
  · rw [norm_zero, norm_le_zero_iff] at h
    /-
      case inl
      𝕜 : Type u_1
      𝕝 : Type u_2
      E : Type u_3
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : NormedRing 𝕝
      inst✝⁴ : NormedSpace 𝕜 𝕝
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : SMulWithZero 𝕝 E
      inst✝ : IsScalarTower 𝕜 𝕝 E
      s : Set E
      hs : Balanced 𝕝 s
      a : 𝕝
      h : Eq a 0
      ⊢ HasSubset.Subset (HSMul.hSMul a s) (HSMul.hSMul 0 s)
    -/
    simp only [h, ← image_smul, zero_smul, Subset.rfl]
    /-
      🎉 no goals
    -/
  · calc
      a • s = b • (b⁻¹ • a) • s := by rw [smul_assoc, smul_inv_smul₀ hb]
      _ ⊆ b • s := smul_set_mono <| hs _ <| by
        rw [norm_smul, norm_inv, ← div_eq_inv_mul]
        exact div_le_one_of_le₀ h (norm_nonneg _)


theorem Balanced.smul_mem_mono [SMulCommClass 𝕝 𝕜 E] (hs : Balanced 𝕝 s) {a : 𝕜} {b : 𝕝}
    (ha : a • x ∈ s) (hba : ‖b‖ ≤ ‖a‖) : b • x ∈ s := by
  /-
    𝕜 : Type u_1
    𝕝 : Type u_2
    E : Type u_3
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : NormedRing 𝕝
    inst✝⁵ : NormedSpace 𝕜 𝕝
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : SMulWithZero 𝕝 E
    inst✝¹ : IsScalarTower 𝕜 𝕝 E
    s : Set E
    x : E
    inst✝ : SMulCommClass 𝕝 𝕜 E
    hs : Balanced 𝕝 s
    a : 𝕜
    b : 𝕝
    ha : Membership.mem s (HSMul.hSMul a x)
    hba : LE.le (Norm.norm b) (Norm.norm a)
    ⊢ Membership.mem s (HSMul.hSMul b x)
  -/
  rcases eq_or_ne a 0 with rfl | ha₀
    /-
      case inl
      𝕜 : Type u_1
      𝕝 : Type u_2
      E : Type u_3
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : NormedRing 𝕝
      inst✝⁵ : NormedSpace 𝕜 𝕝
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : SMulWithZero 𝕝 E
      inst✝¹ : IsScalarTower 𝕜 𝕝 E
      s : Set E
      x : E
      inst✝ : SMulCommClass 𝕝 𝕜 E
      hs : Balanced 𝕝 s
      b : 𝕝
      ha : Membership.mem s (HSMul.hSMul 0 x)
      hba : LE.le (Norm.norm b) (Norm.norm 0)
      ⊢ Membership.mem s (HSMul.hSMul b x)
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  · calc
      (a⁻¹ • b) • a • x ∈ s := by
        refine hs.smul_mem ?_ ha
        rw [norm_smul, norm_inv, ← div_eq_inv_mul]
        exact div_le_one_of_le₀ hba (norm_nonneg _)
      (a⁻¹ • b) • a • x = b • x := by rw [smul_comm, smul_assoc, smul_inv_smul₀ ha₀]


theorem Balanced.subset_smul (hA : Balanced 𝕜 A) (ha : 1 ≤ ‖a‖) : A ⊆ a • A := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    A : Set E
    a : 𝕜
    hA : Balanced 𝕜 A
    ha : LE.le 1 (Norm.norm a)
    ⊢ HasSubset.Subset A (HSMul.hSMul a A)
  -/
  rw [← @norm_one 𝕜] at ha; simpa using hA.smul_mono ha
                            /-
                              🎉 no goals
                            -/


theorem Balanced.smul_congr (hs : Balanced 𝕜 A) (h : ‖a‖ = ‖b‖) : a • A = b • A :=
  (hs.smul_mono h.le).antisymm (hs.smul_mono h.ge)


theorem Balanced.smul_eq (hA : Balanced 𝕜 A) (ha : ‖a‖ = 1) : a • A = A :=
  (hA _ ha.le).antisymm <| hA.subset_smul ha.ge


/-- A balanced set absorbs itself. -/
theorem Balanced.absorbs_self (hA : Balanced 𝕜 A) : Absorbs 𝕜 A A :=
  .of_norm ⟨1, fun _ => hA.subset_smul⟩


theorem Balanced.smul_mem_iff (hs : Balanced 𝕜 s) (h : ‖a‖ = ‖b‖) : a • x ∈ s ↔ b • x ∈ s :=
  ⟨(hs.smul_mem_mono · h.ge), (hs.smul_mem_mono · h.le)⟩


@[deprecated (since := "2024-02-02")] alias Balanced.mem_smul_iff := Balanced.smul_mem_iff


/-- Every neighbourhood of the origin is absorbent. -/
theorem absorbent_nhds_zero (hA : A ∈ 𝓝 (0 : E)) : Absorbent 𝕜 A :=
  absorbent_iff_inv_smul.2 fun x ↦ Filter.tendsto_inv₀_cobounded.smul tendsto_const_nhds <| by
    /-
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      A : Set E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      hA : Membership.mem (nhds 0) A
      x : E
      ⊢ Membership.mem (nhds (HSMul.hSMul 0 x)) A
    -/
    rwa [zero_smul]
    /-
      🎉 no goals
    -/


/-- The union of `{0}` with the interior of a balanced set is balanced. -/
theorem Balanced.zero_insert_interior (hA : Balanced 𝕜 A) :
    Balanced 𝕜 (insert 0 (interior A)) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    A : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul 𝕜 E
    hA : Balanced 𝕜 A
    ⊢ Balanced 𝕜 (Insert.insert 0 (interior A))
  -/
  intro a ha
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    A : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul 𝕜 E
    hA : Balanced 𝕜 A
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    ⊢ HasSubset.Subset (HSMul.hSMul a (Insert.insert 0 (interior A))) (Insert.inse …
  -/
  obtain rfl | h := eq_or_ne a 0
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      A : Set E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      hA : Balanced 𝕜 A
      ha : LE.le (Norm.norm 0) 1
      ⊢ HasSubset.Subset (HSMul.hSMul 0 (Insert.insert 0 (interior A))) (Insert.inse …
    -/
  · rw [zero_smul_set]
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      A : Set E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      hA : Balanced 𝕜 A
      ha : LE.le (Norm.norm 0) 1
      ⊢ HasSubset.Subset 0 (Insert.insert 0 (interior A))
    -/
    exacts [subset_union_left, ⟨0, Or.inl rfl⟩]
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      A : Set E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      hA : Balanced 𝕜 A
      a : 𝕜
      ha : LE.le (Norm.norm a) 1
      h : Ne a 0
      ⊢ HasSubset.Subset (HSMul.hSMul a (Insert.insert 0 (interior A))) (Insert.inse …
    -/
  · rw [← image_smul, image_insert_eq, smul_zero]
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      A : Set E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      hA : Balanced 𝕜 A
      a : 𝕜
      ha : LE.le (Norm.norm a) 1
      h : Ne a 0
      ⊢ HasSubset.Subset (Insert.insert 0 (Set.image (fun x => HSMul.hSMul a x) (int …
    -/
    apply insert_subset_insert
    /-
      case inr.h
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      A : Set E
      inst✝¹ : TopologicalSpace E
      inst✝ : ContinuousSMul 𝕜 E
      hA : Balanced 𝕜 A
      a : 𝕜
      ha : LE.le (Norm.norm a) 1
      h : Ne a 0
      ⊢ HasSubset.Subset (Set.image (fun x => HSMul.hSMul a x) (interior A)) (interi …
    -/
    exact ((isOpenMap_smul₀ h).mapsTo_interior <| hA.smul_mem ha).image_subset
    /-
      🎉 no goals
    -/


@[deprecated Balanced.zero_insert_interior (since := "2024-02-03")]
theorem balanced_zero_union_interior (hA : Balanced 𝕜 A) : Balanced 𝕜 ((0 : Set E) ∪ interior A) :=
  hA.zero_insert_interior


/-- The interior of a balanced set is balanced if it contains the origin. -/
protected theorem Balanced.interior (hA : Balanced 𝕜 A) (h : (0 : E) ∈ interior A) :
    Balanced 𝕜 (interior A) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    A : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul 𝕜 E
    hA : Balanced 𝕜 A
    h : Membership.mem (interior A) 0
    ⊢ Balanced 𝕜 (interior A)
  -/
  rw [← insert_eq_self.2 h]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    A : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul 𝕜 E
    hA : Balanced 𝕜 A
    h : Membership.mem (interior A) 0
    ⊢ Balanced 𝕜 (Insert.insert 0 (interior A))
  -/
  exact hA.zero_insert_interior
  /-
    🎉 no goals
  -/


protected theorem Balanced.closure (hA : Balanced 𝕜 A) : Balanced 𝕜 (closure A) := fun _a ha =>
  (image_closure_subset_closure_image <| continuous_const_smul _).trans <|
    closure_mono <| hA _ ha


@[deprecated Absorbent.zero_mem (since := "2024-02-02")]
theorem Absorbent.zero_mem' (hs : Absorbent 𝕜 s) : (0 : E) ∈ s := hs.zero_mem


protected theorem Balanced.convexHull (hs : Balanced 𝕜 s) : Balanced 𝕜 (convexHull ℝ s) := by
  suffices Convex ℝ { x | ∀ a : 𝕜, ‖a‖ ≤ 1 → a • x ∈ convexHull ℝ s } by
    rw [balanced_iff_smul_mem] at hs ⊢
    refine fun a ha x hx => convexHull_min ?_ this hx a ha
    exact fun y hy a ha => subset_convexHull ℝ s (hs ha hy)
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    s : Set E
    inst✝¹ : Module Real E
    inst✝ : SMulCommClass Real 𝕜 E
    hs : Balanced 𝕜 s
    ⊢ Convex Real (setOf fun x => ∀ (a : 𝕜), LE.le (Norm.norm a) 1 → Membership.me …
  -/
  intro x hx y hy u v hu hv huv a ha
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    s : Set E
    inst✝¹ : Module Real E
    inst✝ : SMulCommClass Real 𝕜 E
    hs : Balanced 𝕜 s
    x : E
    hx : Membership.mem (setOf fun x => ∀ (a : 𝕜), LE.le (Norm.norm a) 1 → Members …
    y : E
    hy : Membership.mem (setOf fun x => ∀ (a : 𝕜), LE.le (Norm.norm a) 1 → Members …
    u v : Real
    hu : LE.le 0 u
    hv : LE.le 0 v
    huv : Eq (HAdd.hAdd u v) 1
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    ⊢ Membership.mem ((convexHull Real) s) (HSMul.hSMul a (HAdd.hAdd (HSMul.hSMul  …
  -/
  simp only [smul_add, ← smul_comm]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    s : Set E
    inst✝¹ : Module Real E
    inst✝ : SMulCommClass Real 𝕜 E
    hs : Balanced 𝕜 s
    x : E
    hx : Membership.mem (setOf fun x => ∀ (a : 𝕜), LE.le (Norm.norm a) 1 → Members …
    y : E
    hy : Membership.mem (setOf fun x => ∀ (a : 𝕜), LE.le (Norm.norm a) 1 → Members …
    u v : Real
    hu : LE.le 0 u
    hv : LE.le 0 v
    huv : Eq (HAdd.hAdd u v) 1
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    ⊢ Membership.mem ((convexHull Real) s) (HAdd.hAdd (HSMul.hSMul u (HSMul.hSMul  …
  -/
  exact convex_convexHull ℝ s (hx a ha) (hy a ha) hu hv huv
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias balanced_convexHull_of_balanced := Balanced.convexHull


theorem balanced_iff_neg_mem (hs : Convex ℝ s) : Balanced ℝ s ↔ ∀ ⦃x⦄, x ∈ s → -x ∈ s := by
  /-
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    hs : Convex Real s
    ⊢ Iff (Balanced Real s) (∀ ⦃x : E⦄, Membership.mem s x → Membership.mem s (Neg …
  -/
  refine ⟨fun h x => h.neg_mem_iff.2, fun h a ha => smul_set_subset_iff.2 fun x hx => ?_⟩
  /-
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    hs : Convex Real s
    h : ∀ ⦃x : E⦄, Membership.mem s x → Membership.mem s (Neg.neg x)
    a : Real
    ha : LE.le (Norm.norm a) 1
    x : E
    hx : Membership.mem s x
    ⊢ Membership.mem s (HSMul.hSMul a x)
  -/
  rw [Real.norm_eq_abs, abs_le] at ha
  /-
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    hs : Convex Real s
    h : ∀ ⦃x : E⦄, Membership.mem s x → Membership.mem s (Neg.neg x)
    a : Real
    ha : And (LE.le (-1) a) (LE.le a 1)
    x : E
    hx : Membership.mem s x
    ⊢ Membership.mem s (HSMul.hSMul a x)
  -/
  rw [show a = -((1 - a) / 2) + (a - -1) / 2 by ring, add_smul, neg_smul, ← smul_neg]
  exact hs (h hx) hx (div_nonneg (sub_nonneg_of_le ha.2) zero_le_two)
    (div_nonneg (sub_nonneg_of_le ha.1) zero_le_two) (by ring)


