@[to_additive (attr := simp)] lemma csSup_one : sSup (1 : Set M) = 1 := csSup_singleton _

@[to_additive (attr := simp)] lemma csInf_one : sInf (1 : Set M) = 1 := csInf_singleton _


@[to_additive]
lemma csSup_inv (hs₀ : s.Nonempty) (hs₁ : BddBelow s) : sSup s⁻¹ = (sInf s)⁻¹ := by
  /-
    M : Type u_1
    inst✝³ : ConditionallyCompleteLattice M
    inst✝² : Group M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    s : Set M
    hs₀ : s.Nonempty
    hs₁ : BddBelow s
    ⊢ Eq (SupSet.sSup (Inv.inv s)) (Inv.inv (InfSet.sInf s))
  -/
  rw [← image_inv_eq_inv]
  /-
    M : Type u_1
    inst✝³ : ConditionallyCompleteLattice M
    inst✝² : Group M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    s : Set M
    hs₀ : s.Nonempty
    hs₁ : BddBelow s
    ⊢ Eq (SupSet.sSup (Set.image (fun x => Inv.inv x) s)) (Inv.inv (InfSet.sInf s))
  -/
  exact ((OrderIso.inv _).map_csInf' hs₀ hs₁).symm
  /-
    🎉 no goals
  -/


@[to_additive]
lemma csInf_inv (hs₀ : s.Nonempty) (hs₁ : BddAbove s) : sInf s⁻¹ = (sSup s)⁻¹ := by
  /-
    M : Type u_1
    inst✝³ : ConditionallyCompleteLattice M
    inst✝² : Group M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    s : Set M
    hs₀ : s.Nonempty
    hs₁ : BddAbove s
    ⊢ Eq (InfSet.sInf (Inv.inv s)) (Inv.inv (SupSet.sSup s))
  -/
  rw [← image_inv_eq_inv]
  /-
    M : Type u_1
    inst✝³ : ConditionallyCompleteLattice M
    inst✝² : Group M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    s : Set M
    hs₀ : s.Nonempty
    hs₁ : BddAbove s
    ⊢ Eq (InfSet.sInf (Set.image (fun x => Inv.inv x) s)) (Inv.inv (SupSet.sSup s))
  -/
  exact ((OrderIso.inv _).map_csSup' hs₀ hs₁).symm
  /-
    🎉 no goals
  -/


@[to_additive]
lemma csSup_mul (hs₀ : s.Nonempty) (hs₁ : BddAbove s) (ht₀ : t.Nonempty) (ht₁ : BddAbove t) :
    sSup (s * t) = sSup s * sSup t :=
  csSup_image2_eq_csSup_csSup (fun _ => (OrderIso.mulRight _).to_galoisConnection)
    (fun _ => (OrderIso.mulLeft _).to_galoisConnection) hs₀ hs₁ ht₀ ht₁


@[to_additive]
lemma csInf_mul (hs₀ : s.Nonempty) (hs₁ : BddBelow s) (ht₀ : t.Nonempty) (ht₁ : BddBelow t) :
    sInf (s * t) = sInf s * sInf t :=
  csInf_image2_eq_csInf_csInf (fun _ => (OrderIso.mulRight _).symm.to_galoisConnection)
    (fun _ => (OrderIso.mulLeft _).symm.to_galoisConnection) hs₀ hs₁ ht₀ ht₁


@[to_additive]
lemma csSup_div (hs₀ : s.Nonempty) (hs₁ : BddAbove s) (ht₀ : t.Nonempty) (ht₁ : BddBelow t) :
    sSup (s / t) = sSup s / sInf t := by
  /-
    M : Type u_1
    inst✝³ : ConditionallyCompleteLattice M
    inst✝² : Group M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    s t : Set M
    hs₀ : s.Nonempty
    hs₁ : BddAbove s
    ht₀ : t.Nonempty
    ht₁ : BddBelow t
    ⊢ Eq (SupSet.sSup (HDiv.hDiv s t)) (HDiv.hDiv (SupSet.sSup s) (InfSet.sInf t))
  -/
  rw [div_eq_mul_inv, csSup_mul hs₀ hs₁ ht₀.inv ht₁.inv, csSup_inv ht₀ ht₁, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma csInf_div (hs₀ : s.Nonempty) (hs₁ : BddBelow s) (ht₀ : t.Nonempty) (ht₁ : BddAbove t) :
    sInf (s / t) = sInf s / sSup t := by
  /-
    M : Type u_1
    inst✝³ : ConditionallyCompleteLattice M
    inst✝² : Group M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    s t : Set M
    hs₀ : s.Nonempty
    hs₁ : BddBelow s
    ht₀ : t.Nonempty
    ht₁ : BddAbove t
    ⊢ Eq (InfSet.sInf (HDiv.hDiv s t)) (HDiv.hDiv (InfSet.sInf s) (SupSet.sSup t))
  -/
  rw [div_eq_mul_inv, csInf_mul hs₀ hs₁ ht₀.inv ht₁.inv, csInf_inv ht₀ ht₁, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[to_additive] lemma sSup_one : sSup (1 : Set M) = 1 := sSup_singleton

@[to_additive] lemma sInf_one : sInf (1 : Set M) = 1 := sInf_singleton


@[to_additive]
lemma sSup_inv (s : Set M) : sSup s⁻¹ = (sInf s)⁻¹ := by
  /-
    M : Type u_1
    inst✝³ : CompleteLattice M
    inst✝² : Group M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    s : Set M
    ⊢ Eq (SupSet.sSup (Inv.inv s)) (Inv.inv (InfSet.sInf s))
  -/
  rw [← image_inv_eq_inv, sSup_image]
  /-
    M : Type u_1
    inst✝³ : CompleteLattice M
    inst✝² : Group M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    s : Set M
    ⊢ Eq (iSup fun a => iSup fun h => Inv.inv a) (Inv.inv (InfSet.sInf s))
  -/
  exact ((OrderIso.inv M).map_sInf _).symm
  /-
    🎉 no goals
  -/


@[to_additive]
lemma sInf_inv (s : Set M) : sInf s⁻¹ = (sSup s)⁻¹ := by
  /-
    M : Type u_1
    inst✝³ : CompleteLattice M
    inst✝² : Group M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    s : Set M
    ⊢ Eq (InfSet.sInf (Inv.inv s)) (Inv.inv (SupSet.sSup s))
  -/
  rw [← image_inv_eq_inv, sInf_image]
  /-
    M : Type u_1
    inst✝³ : CompleteLattice M
    inst✝² : Group M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    s : Set M
    ⊢ Eq (iInf fun a => iInf fun h => Inv.inv a) (Inv.inv (SupSet.sSup s))
  -/
  exact ((OrderIso.inv M).map_sSup _).symm
  /-
    🎉 no goals
  -/


@[to_additive]
lemma sSup_mul : sSup (s * t) = sSup s * sSup t :=
  (sSup_image2_eq_sSup_sSup fun _ => (OrderIso.mulRight _).to_galoisConnection) fun _ =>
    (OrderIso.mulLeft _).to_galoisConnection


@[to_additive]
lemma sInf_mul : sInf (s * t) = sInf s * sInf t :=
  (sInf_image2_eq_sInf_sInf fun _ => (OrderIso.mulRight _).symm.to_galoisConnection) fun _ =>
    (OrderIso.mulLeft _).symm.to_galoisConnection


@[to_additive]
                                                      /-
                                                        M : Type u_1
                                                        inst✝³ : CompleteLattice M
                                                        inst✝² : Group M
                                                        inst✝¹ : MulLeftMono M
                                                        inst✝ : MulRightMono M
                                                        s t : Set M
                                                        ⊢ Eq (SupSet.sSup (HDiv.hDiv s t)) (HDiv.hDiv (SupSet.sSup s) (InfSet.sInf t))
                                                      -/
lemma sSup_div : sSup (s / t) = sSup s / sInf t := by simp_rw [div_eq_mul_inv, sSup_mul, sSup_inv]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive]
                                                      /-
                                                        M : Type u_1
                                                        inst✝³ : CompleteLattice M
                                                        inst✝² : Group M
                                                        inst✝¹ : MulLeftMono M
                                                        inst✝ : MulRightMono M
                                                        s t : Set M
                                                        ⊢ Eq (InfSet.sInf (HDiv.hDiv s t)) (HDiv.hDiv (InfSet.sInf s) (SupSet.sSup t))
                                                      -/
lemma sInf_div : sInf (s / t) = sInf s / sSup t := by simp_rw [div_eq_mul_inv, sInf_mul, sInf_inv]
                                                      /-
                                                        🎉 no goals
                                                      -/


