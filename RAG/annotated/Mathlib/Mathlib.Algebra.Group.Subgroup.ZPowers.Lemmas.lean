theorem range_zpowersHom (g : G) : (zpowersHom G g).range = zpowers g := rfl


@[to_additive]
instance (a : G) : Countable (zpowers a) := Set.surjective_onto_range.countable


@[simp]
theorem range_zmultiplesHom (a : A) : (zmultiplesHom A a).range = zmultiples a :=
  rfl


@[simp]
theorem intCast_mul_mem_zmultiples : ↑(k : ℤ) * r ∈ zmultiples r := by
  /-
    R : Type u_4
    inst✝ : Ring R
    r : R
    k : Int
    ⊢ Membership.mem (AddSubgroup.zmultiples r) (HMul.hMul (↑k) r)
  -/
  simpa only [← zsmul_eq_mul] using zsmul_mem_zmultiples r k
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias int_cast_mul_mem_zmultiples := intCast_mul_mem_zmultiples


@[simp]
theorem intCast_mem_zmultiples_one : ↑(k : ℤ) ∈ zmultiples (1 : R) :=
                               /-
                                 R : Type u_4
                                 inst✝ : Ring R
                                 k : Int
                                 ⊢ Eq ((fun x => HSMul.hSMul x 1) k) ↑k
                               -/
  mem_zmultiples_iff.mp ⟨k, by simp⟩
                               /-
                                 🎉 no goals
                               -/


@[deprecated (since := "2024-04-17")]
alias int_cast_mem_zmultiples_one := intCast_mem_zmultiples_one


@[simp] lemma Int.range_castAddHom {A : Type*} [AddGroupWithOne A] :
    (Int.castAddHom A).range = AddSubgroup.zmultiples 1 := by
  /-
    A : Type u_4
    inst✝ : AddGroupWithOne A
    ⊢ Eq (Int.castAddHom A).range (AddSubgroup.zmultiples 1)
  -/
  ext a
  /-
    case h
    A : Type u_4
    inst✝ : AddGroupWithOne A
    a : A
    ⊢ Iff (Membership.mem (Int.castAddHom A).range a) (Membership.mem (AddSubgroup …
  -/
  simp_rw [AddMonoidHom.mem_range, Int.coe_castAddHom, AddSubgroup.mem_zmultiples_iff, zsmul_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem centralizer_closure (S : Set G) :
    centralizer (closure S : Set G) = ⨅ g ∈ S, centralizer (zpowers g : Set G) :=
  le_antisymm
      (le_iInf fun _ => le_iInf fun hg => centralizer_le <| zpowers_le.2 <| subset_closure hg) <|
    le_centralizer_iff.1 <|
      (closure_le _).2 fun g =>
        SetLike.mem_coe.2 ∘ zpowers_le.1 ∘ le_centralizer_iff.1 ∘ iInf_le_of_le g ∘ iInf_le _


@[to_additive]
theorem center_eq_iInf (S : Set G) (hS : closure S = ⊤) :
    center G = ⨅ g ∈ S, centralizer (zpowers g) := by
  /-
    G : Type u_1
    inst✝ : Group G
    S : Set G
    hS : Eq (Subgroup.closure S) Top.top
    ⊢ Eq (Subgroup.center G) (iInf fun g => iInf fun h => Subgroup.centralizer ↑(S …
  -/
  rw [← centralizer_univ, ← coe_top, ← hS, centralizer_closure]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem center_eq_infi' (S : Set G) (hS : closure S = ⊤) :
    center G = ⨅ g : S, centralizer (zpowers (g : G)) := by
  /-
    G : Type u_1
    inst✝ : Group G
    S : Set G
    hS : Eq (Subgroup.closure S) Top.top
    ⊢ Eq (Subgroup.center G) (iInf fun g => Subgroup.centralizer ↑(Subgroup.zpower …
  -/
  rw [center_eq_iInf S hS, ← iInf_subtype'']
  /-
    🎉 no goals
  -/


