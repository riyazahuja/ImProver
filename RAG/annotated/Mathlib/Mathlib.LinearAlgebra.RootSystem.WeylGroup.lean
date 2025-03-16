/-- The `Weyl group` of a root pairing is the group of automorphisms of the root pairing generated
by reflections. -/
def weylGroup : Subgroup (Aut P) :=
  Subgroup.closure (range (Equiv.reflection P))


lemma reflection_mem_weylGroup : Equiv.reflection P i ∈ P.weylGroup :=
  Subgroup.subset_closure <| mem_range_self i


lemma range_weylGroup_weightHom :
    MonoidHom.range ((Equiv.weightHom P).restrict P.weylGroup) =
      Subgroup.closure (range P.reflection) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Eq ((RootPairing.Equiv.weightHom P).restrict P.weylGroup).range (Subgroup.cl …
  -/
  refine (Subgroup.closure_eq_of_le _ ?_ ?_).symm
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      ⊢ HasSubset.Subset (Set.range P.reflection) ↑((RootPairing.Equiv.weightHom P). …
    -/
  · rintro - ⟨i, rfl⟩
    simp only [MonoidHom.restrict_range, Subgroup.coe_map, Equiv.weightHom_apply, mem_image,
      SetLike.mem_coe]
    /-
      case refine_1.intro
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i : ι
      ⊢ Exists fun x => And (Membership.mem P.weylGroup x) (Eq (RootPairing.Equiv.we …
    -/
    use Equiv.reflection P i
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i : ι
      ⊢ And (Membership.mem P.weylGroup (RootPairing.Equiv.reflection P i)) (Eq (Roo …
    -/
    exact ⟨reflection_mem_weylGroup P i, Equiv.reflection_weightEquiv P i⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      ⊢ LE.le ((RootPairing.Equiv.weightHom P).restrict P.weylGroup).range (Subgroup …
    -/
  · rintro fg ⟨⟨w, hw⟩, rfl⟩
    induction hw using Subgroup.closure_induction'' with
    | one =>
      change ((Equiv.weightHom P).restrict P.weylGroup) 1 ∈ _
      simpa only [map_one] using Subgroup.one_mem _
    | mem w' hw' =>
      obtain ⟨i, rfl⟩ := hw'
      simp only [MonoidHom.restrict_apply, Equiv.weightHom_apply, Equiv.reflection_weightEquiv]
      simpa only [reflection_mem_weylGroup] using Subgroup.subset_closure (mem_range_self i)
    | inv_mem w' hw' =>
      obtain ⟨i, rfl⟩ := hw'
      simp only [Equiv.reflection_inv, MonoidHom.restrict_apply, Equiv.weightHom_apply,
        Equiv.reflection_weightEquiv]
      simpa only [reflection_mem_weylGroup] using Subgroup.subset_closure (mem_range_self i)
    | mul w₁ w₂ hw₁ hw₂ h₁ h₂ =>
      simpa only [← Submonoid.mk_mul_mk _ w₁ w₂ hw₁ hw₂, map_mul] using Subgroup.mul_mem _ h₁ h₂


lemma range_weylGroup_coweightHom :
    MonoidHom.range ((Equiv.coweightHom P).restrict P.weylGroup) =
      Subgroup.closure (range (MulOpposite.op ∘ P.coreflection)) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Eq ((RootPairing.Equiv.coweightHom P).restrict P.weylGroup).range (Subgroup. …
  -/
  refine (Subgroup.closure_eq_of_le _ ?_ ?_).symm
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      ⊢ HasSubset.Subset (Set.range (Function.comp MulOpposite.op P.coreflection)) ↑ …
    -/
  · rintro - ⟨i, rfl⟩
    simp only [MonoidHom.restrict_range, Subgroup.coe_map, Equiv.weightHom_apply, mem_image,
      SetLike.mem_coe]
    /-
      case refine_1.intro
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i : ι
      ⊢ Exists fun x => And (Membership.mem P.weylGroup x) (Eq ((RootPairing.Equiv.c …
    -/
    use Equiv.reflection P i
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i : ι
      ⊢ And (Membership.mem P.weylGroup (RootPairing.Equiv.reflection P i)) (Eq ((Ro …
    -/
    refine ⟨reflection_mem_weylGroup P i, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      ⊢ LE.le ((RootPairing.Equiv.coweightHom P).restrict P.weylGroup).range (Subgro …
    -/
  · rintro fg ⟨⟨w, hw⟩, rfl⟩
    induction hw using Subgroup.closure_induction'' with
    | one =>
      change ((Equiv.coweightHom P).restrict P.weylGroup) 1 ∈ _
      simpa only [map_one] using Subgroup.one_mem _
    | mem w' hw' =>
      obtain ⟨i, rfl⟩ := hw'
      simp only [MonoidHom.restrict_apply, Equiv.coweightHom_apply, Equiv.reflection_coweightEquiv]
      simpa only [reflection_mem_weylGroup] using Subgroup.subset_closure (mem_range_self i)
    | inv_mem w' hw' =>
      obtain ⟨i, rfl⟩ := hw'
      simp only [Equiv.reflection_inv, MonoidHom.restrict_apply, Equiv.coweightHom_apply,
        Equiv.reflection_coweightEquiv]
      simpa only [reflection_mem_weylGroup] using Subgroup.subset_closure (mem_range_self i)
    | mul w₁ w₂ hw₁ hw₂ h₁ h₂ =>
      simpa only [← Submonoid.mk_mul_mk _ w₁ w₂ hw₁ hw₂, map_mul] using Subgroup.mul_mem _ h₁ h₂


/-- The permutation representation of the Weyl group induced by `reflection_perm`. -/
abbrev weylGroupToPerm := (Equiv.indexHom P).restrict P.weylGroup


lemma range_weylGroupToPerm :
    P.weylGroupToPerm.range = Subgroup.closure (range P.reflection_perm) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Eq P.weylGroupToPerm.range (Subgroup.closure (Set.range P.reflection_perm))
  -/
  refine (Subgroup.closure_eq_of_le _ ?_ ?_).symm
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      ⊢ HasSubset.Subset (Set.range P.reflection_perm) ↑P.weylGroupToPerm.range
    -/
  · rintro - ⟨i, rfl⟩
    simp only [MonoidHom.restrict_range, Subgroup.coe_map, Equiv.weightHom_apply, mem_image,
      SetLike.mem_coe]
    /-
      case refine_1.intro
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i : ι
      ⊢ Exists fun x => And (Membership.mem P.weylGroup x) (Eq ((RootPairing.Equiv.i …
    -/
    use Equiv.reflection P i
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i : ι
      ⊢ And (Membership.mem P.weylGroup (RootPairing.Equiv.reflection P i)) (Eq ((Ro …
    -/
    refine ⟨reflection_mem_weylGroup P i, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      ⊢ LE.le P.weylGroupToPerm.range (Subgroup.closure (Set.range P.reflection_perm))
    -/
  · rintro fg ⟨⟨w, hw⟩, rfl⟩
    induction hw using Subgroup.closure_induction'' with
    | one =>
      change ((Equiv.indexHom P).restrict P.weylGroup) 1 ∈ _
      simpa only [map_one] using Subgroup.one_mem _
    | mem w' hw' =>
      obtain ⟨i, rfl⟩ := hw'
      simp only [MonoidHom.restrict_apply, Equiv.indexHom_apply, Equiv.reflection_indexEquiv]
      simpa only [reflection_mem_weylGroup] using Subgroup.subset_closure (mem_range_self i)
    | inv_mem w' hw' =>
      obtain ⟨i, rfl⟩ := hw'
      simp only [Equiv.reflection_inv, MonoidHom.restrict_apply, Equiv.indexHom_apply,
        Equiv.reflection_indexEquiv]
      simpa only [reflection_mem_weylGroup] using Subgroup.subset_closure (mem_range_self i)
    | mul w₁ w₂ hw₁ hw₂ h₁ h₂ =>
      simpa only [← Submonoid.mk_mul_mk _ w₁ w₂ hw₁ hw₂, map_mul] using Subgroup.mul_mem _ h₁ h₂


