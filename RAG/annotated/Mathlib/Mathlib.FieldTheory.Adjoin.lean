/-- `adjoin F S` extends a field `F` by adjoining a set `S ⊆ E`. -/
@[stacks 09FZ "first part"]
def adjoin : IntermediateField F E :=
  { Subfield.closure (Set.range (algebraMap F E) ∪ S) with
    algebraMap_mem' := fun x => Subfield.subset_closure (Or.inl (Set.mem_range_self x)) }


@[simp]
theorem adjoin_toSubfield :
    (adjoin F S).toSubfield = Subfield.closure (Set.range (algebraMap F E) ∪ S) := rfl


theorem mem_adjoin_range_iff {ι : Type*} (i : ι → E) (x : E) :
    x ∈ adjoin F (Set.range i) ↔ ∃ r s : MvPolynomial ι F,
      x = MvPolynomial.aeval i r / MvPolynomial.aeval i s := by
  simp_rw [adjoin, mem_mk, Subring.mem_toSubsemiring, Subfield.mem_toSubring,
    Subfield.mem_closure_iff, ← Algebra.adjoin_eq_ring_closure, Subalgebra.mem_toSubring,
    Algebra.adjoin_range_eq_range_aeval, AlgHom.mem_range, exists_exists_eq_and]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ι : Type u_3
    i : ι → E
    x : E
    ⊢ Iff (Exists fun a => Exists fun a_1 => Eq (HDiv.hDiv ((MvPolynomial.aeval i) …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem mem_adjoin_iff (x : E) :
    x ∈ adjoin F S ↔ ∃ r s : MvPolynomial S F,
      x = MvPolynomial.aeval Subtype.val r / MvPolynomial.aeval Subtype.val s := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    x : E
    ⊢ Iff (Membership.mem (IntermediateField.adjoin F S) x) (Exists fun r => Exist …
  -/
  rw [← mem_adjoin_range_iff, Subtype.range_coe]
  /-
    🎉 no goals
  -/


theorem mem_adjoin_simple_iff {α : E} (x : E) :
    x ∈ adjoin F {α} ↔ ∃ r s : F[X], x = aeval α r / aeval α s := by
  simp only [adjoin, mem_mk, Subring.mem_toSubsemiring, Subfield.mem_toSubring,
    Subfield.mem_closure_iff, ← Algebra.adjoin_eq_ring_closure, Subalgebra.mem_toSubring,
    Algebra.adjoin_singleton_eq_range_aeval, AlgHom.mem_range, exists_exists_eq_and]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α x : E
    ⊢ Iff (Exists fun a => Exists fun a_1 => Eq (HDiv.hDiv ((Polynomial.aeval α) a …
  -/
  tauto
  /-
    🎉 no goals
  -/


@[simp]
theorem adjoin_le_iff {S : Set E} {T : IntermediateField F E} : adjoin F S ≤ T ↔ S ⊆ T :=
  ⟨fun H => le_trans (le_trans Set.subset_union_right Subfield.subset_closure) H, fun H =>
    (@Subfield.closure_le E _ (Set.range (algebraMap F E) ∪ S) T.toSubfield).mpr
      (Set.union_subset (IntermediateField.set_range_subset T) H)⟩


theorem gc : GaloisConnection (adjoin F : Set E → IntermediateField F E)
    (fun (x : IntermediateField F E) => (x : Set E)) := fun _ _ =>
  adjoin_le_iff


/-- Galois insertion between `adjoin` and `coe`. -/
def gi : GaloisInsertion (adjoin F : Set E → IntermediateField F E)
    (fun (x : IntermediateField F E) => (x : Set E)) where
  choice s hs := (adjoin F s).copy s <| le_antisymm (gc.le_u_l s) hs
  gc := IntermediateField.gc
  le_l_u S := (IntermediateField.gc (S : Set E) (adjoin F S)).1 <| le_rfl
  choice_eq _ _ := copy_eq _ _ _


instance : CompleteLattice (IntermediateField F E) where
  __ := GaloisInsertion.liftCompleteLattice IntermediateField.gi
  bot :=
    { toSubalgebra := ⊥
                     /-
                       F : Type u_1
                       inst✝² : Field F
                       E : Type u_2
                       inst✝¹ : Field E
                       inst✝ : Algebra F E
                       ⊢ ∀ (x : E), Membership.mem Bot.bot.carrier x → Membership.mem Bot.bot.carrier …
                     -/
      inv_mem' := by rintro x ⟨r, rfl⟩; exact ⟨r⁻¹, map_inv₀ _ _⟩ }
                                        /-
                                          🎉 no goals
                                        -/
  bot_le x := (bot_le : ⊥ ≤ x.toSubalgebra)


theorem sup_def (S T : IntermediateField F E) : S ⊔ T = adjoin F (S ∪ T : Set E) := rfl


theorem sSup_def (S : Set (IntermediateField F E)) :
    sSup S = adjoin F (⋃₀ (SetLike.coe '' S)) := rfl


instance : Inhabited (IntermediateField F E) :=
  ⟨⊤⟩


instance : Unique (IntermediateField F F) :=
  { inferInstanceAs (Inhabited (IntermediateField F F)) with
    uniq := fun _ ↦ toSubalgebra_injective <| Subsingleton.elim _ _ }


theorem coe_bot : ↑(⊥ : IntermediateField F E) = Set.range (algebraMap F E) := rfl


theorem mem_bot {x : E} : x ∈ (⊥ : IntermediateField F E) ↔ x ∈ Set.range (algebraMap F E) :=
  Iff.rfl


@[simp]
theorem bot_toSubalgebra : (⊥ : IntermediateField F E).toSubalgebra = ⊥ := rfl


theorem bot_toSubfield : (⊥ : IntermediateField F E).toSubfield = (algebraMap F E).fieldRange :=
  rfl


@[simp]
theorem coe_top : ↑(⊤ : IntermediateField F E) = (Set.univ : Set E) :=
  rfl


@[simp]
theorem mem_top {x : E} : x ∈ (⊤ : IntermediateField F E) :=
  trivial


@[simp]
theorem top_toSubalgebra : (⊤ : IntermediateField F E).toSubalgebra = ⊤ :=
  rfl


@[simp]
theorem top_toSubfield : (⊤ : IntermediateField F E).toSubfield = ⊤ :=
  rfl


@[simp, norm_cast]
theorem coe_inf (S T : IntermediateField F E) : (↑(S ⊓ T) : Set E) = (S : Set E) ∩ T :=
  rfl


@[simp]
theorem mem_inf {S T : IntermediateField F E} {x : E} : x ∈ S ⊓ T ↔ x ∈ S ∧ x ∈ T :=
  Iff.rfl


@[simp]
theorem inf_toSubalgebra (S T : IntermediateField F E) :
    (S ⊓ T).toSubalgebra = S.toSubalgebra ⊓ T.toSubalgebra :=
  rfl


@[simp]
theorem inf_toSubfield (S T : IntermediateField F E) :
    (S ⊓ T).toSubfield = S.toSubfield ⊓ T.toSubfield :=
  rfl


@[simp]
theorem sup_toSubfield (S T : IntermediateField F E) :
    (S ⊔ T).toSubfield = S.toSubfield ⊔ T.toSubfield := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S T : IntermediateField F E
    ⊢ Eq (Max.max S T).toSubfield (Max.max S.toSubfield T.toSubfield)
  -/
  rw [← S.toSubfield.closure_eq, ← T.toSubfield.closure_eq, ← Subfield.closure_union]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S T : IntermediateField F E
    ⊢ Eq (Max.max S T).toSubfield (Subfield.closure (Union.union ↑S.toSubfield ↑T. …
  -/
  simp_rw [sup_def, adjoin_toSubfield, coe_toSubfield]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S T : IntermediateField F E
    ⊢ Eq (Subfield.closure (Union.union (Set.range ⇑(algebraMap F E)) (Union.union …
  -/
  congr 1
  /-
    case e_s
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S T : IntermediateField F E
    ⊢ Eq (Union.union (Set.range ⇑(algebraMap F E)) (Union.union ↑S ↑T)) (Union.un …
  -/
  rw [Set.union_eq_right]
  /-
    case e_s
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S T : IntermediateField F E
    ⊢ HasSubset.Subset (Set.range ⇑(algebraMap F E)) (Union.union ↑S ↑T)
  -/
  rintro _ ⟨x, rfl⟩
  /-
    case e_s.intro
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S T : IntermediateField F E
    x : F
    ⊢ Membership.mem (Union.union ↑S ↑T) ((algebraMap F E) x)
  -/
  exact Set.mem_union_left _ (algebraMap_mem S x)
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_sInf (S : Set (IntermediateField F E)) : (↑(sInf S) : Set E) =
    sInf ((fun (x : IntermediateField F E) => (x : Set E)) '' S) :=
  rfl


@[simp]
theorem sInf_toSubalgebra (S : Set (IntermediateField F E)) :
    (sInf S).toSubalgebra = sInf (toSubalgebra '' S) :=
                              /-
                                F : Type u_1
                                inst✝² : Field F
                                E : Type u_2
                                inst✝¹ : Field E
                                inst✝ : Algebra F E
                                S : Set (IntermediateField F E)
                                ⊢ Eq ↑(InfSet.sInf S).toSubalgebra ↑(InfSet.sInf (Set.image IntermediateField. …
                              -/
  SetLike.coe_injective <| by simp [Set.sUnion_image]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem sInf_toSubfield (S : Set (IntermediateField F E)) :
    (sInf S).toSubfield = sInf (toSubfield '' S) :=
                              /-
                                F : Type u_1
                                inst✝² : Field F
                                E : Type u_2
                                inst✝¹ : Field E
                                inst✝ : Algebra F E
                                S : Set (IntermediateField F E)
                                ⊢ Eq ↑(InfSet.sInf S).toSubfield ↑(InfSet.sInf (Set.image IntermediateField.to …
                              -/
  SetLike.coe_injective <| by simp [Set.sUnion_image]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem sSup_toSubfield (S : Set (IntermediateField F E)) (hS : S.Nonempty) :
    (sSup S).toSubfield = sSup (toSubfield '' S) := by
  have h : toSubfield '' S = Subfield.closure '' (SetLike.coe '' S) := by
    rw [Set.image_image]
    congr! with x
    exact x.toSubfield.closure_eq.symm
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set (IntermediateField F E)
    hS : S.Nonempty
    h : Eq (Set.image IntermediateField.toSubfield S) (Set.image Subfield.closure  …
    ⊢ Eq (SupSet.sSup S).toSubfield (SupSet.sSup (Set.image IntermediateField.toSu …
  -/
  rw [h, sSup_image, ← Subfield.closure_sUnion, sSup_def, adjoin_toSubfield]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set (IntermediateField F E)
    hS : S.Nonempty
    h : Eq (Set.image IntermediateField.toSubfield S) (Set.image Subfield.closure  …
    ⊢ Eq (Subfield.closure (Union.union (Set.range ⇑(algebraMap F E)) (Set.image S …
  -/
  congr 1
  /-
    case e_s
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set (IntermediateField F E)
    hS : S.Nonempty
    h : Eq (Set.image IntermediateField.toSubfield S) (Set.image Subfield.closure  …
    ⊢ Eq (Union.union (Set.range ⇑(algebraMap F E)) (Set.image SetLike.coe S).sUni …
  -/
  rw [Set.union_eq_right]
  /-
    case e_s
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set (IntermediateField F E)
    hS : S.Nonempty
    h : Eq (Set.image IntermediateField.toSubfield S) (Set.image Subfield.closure  …
    ⊢ HasSubset.Subset (Set.range ⇑(algebraMap F E)) (Set.image SetLike.coe S).sUn …
  -/
  rintro _ ⟨x, rfl⟩
  /-
    case e_s.intro
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set (IntermediateField F E)
    hS : S.Nonempty
    h : Eq (Set.image IntermediateField.toSubfield S) (Set.image Subfield.closure  …
    x : F
    ⊢ Membership.mem (Set.image SetLike.coe S).sUnion ((algebraMap F E) x)
  -/
  obtain ⟨y, hy⟩ := hS
  /-
    case e_s.intro.intro
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set (IntermediateField F E)
    h : Eq (Set.image IntermediateField.toSubfield S) (Set.image Subfield.closure  …
    x : F
    y : IntermediateField F E
    hy : Membership.mem S y
    ⊢ Membership.mem (Set.image SetLike.coe S).sUnion ((algebraMap F E) x)
  -/
  simp only [Set.mem_sUnion, Set.mem_image, exists_exists_and_eq_and, SetLike.mem_coe]
  /-
    case e_s.intro.intro
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set (IntermediateField F E)
    h : Eq (Set.image IntermediateField.toSubfield S) (Set.image Subfield.closure  …
    x : F
    y : IntermediateField F E
    hy : Membership.mem S y
    ⊢ Exists fun a => And (Membership.mem S a) (Membership.mem a ((algebraMap F E) …
  -/
  exact ⟨y, hy, algebraMap_mem y x⟩
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_iInf {ι : Sort*} (S : ι → IntermediateField F E) : (↑(iInf S) : Set E) = ⋂ i, S i := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ι : Sort u_3
    S : ι → IntermediateField F E
    ⊢ Eq (↑(iInf S)) (Set.iInter fun i => ↑(S i))
  -/
  simp [iInf]
  /-
    🎉 no goals
  -/


@[simp]
theorem iInf_toSubalgebra {ι : Sort*} (S : ι → IntermediateField F E) :
    (iInf S).toSubalgebra = ⨅ i, (S i).toSubalgebra :=
                              /-
                                F : Type u_1
                                inst✝² : Field F
                                E : Type u_2
                                inst✝¹ : Field E
                                inst✝ : Algebra F E
                                ι : Sort u_3
                                S : ι → IntermediateField F E
                                ⊢ Eq ↑(iInf S).toSubalgebra ↑(iInf fun i => (S i).toSubalgebra)
                              -/
  SetLike.coe_injective <| by simp [iInf]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem iInf_toSubfield {ι : Sort*} (S : ι → IntermediateField F E) :
    (iInf S).toSubfield = ⨅ i, (S i).toSubfield :=
                              /-
                                F : Type u_1
                                inst✝² : Field F
                                E : Type u_2
                                inst✝¹ : Field E
                                inst✝ : Algebra F E
                                ι : Sort u_3
                                S : ι → IntermediateField F E
                                ⊢ Eq ↑(iInf S).toSubfield ↑(iInf fun i => (S i).toSubfield)
                              -/
  SetLike.coe_injective <| by simp [iInf]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem iSup_toSubfield {ι : Sort*} [Nonempty ι] (S : ι → IntermediateField F E) :
    (iSup S).toSubfield = ⨆ i, (S i).toSubfield := by
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    ι : Sort u_3
    inst✝ : Nonempty ι
    S : ι → IntermediateField F E
    ⊢ Eq (iSup S).toSubfield (iSup fun i => (S i).toSubfield)
  -/
  simp only [iSup, Set.range_nonempty, sSup_toSubfield, ← Set.range_comp, Function.comp_def]
  /-
    🎉 no goals
  -/


/-- Construct an algebra isomorphism from an equality of intermediate fields -/
@[simps! apply]
def equivOfEq {S T : IntermediateField F E} (h : S = T) : S ≃ₐ[F] T :=
  Subalgebra.equivOfEq _ _ (congr_arg toSubalgebra h)


@[simp]
theorem equivOfEq_symm {S T : IntermediateField F E} (h : S = T) :
    (equivOfEq h).symm = equivOfEq h.symm :=
  rfl


@[simp]
theorem equivOfEq_rfl (S : IntermediateField F E) : equivOfEq (rfl : S = S) = AlgEquiv.refl := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : IntermediateField F E
    ⊢ Eq (IntermediateField.equivOfEq ⋯) AlgEquiv.refl
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


@[simp]
theorem equivOfEq_trans {S T U : IntermediateField F E} (hST : S = T) (hTU : T = U) :
    (equivOfEq hST).trans (equivOfEq hTU) = equivOfEq (hST.trans hTU) :=
  rfl


/-- The bottom intermediate_field is isomorphic to the field. -/
noncomputable def botEquiv : (⊥ : IntermediateField F E) ≃ₐ[F] F :=
  (Subalgebra.equivOfEq _ _ bot_toSubalgebra).trans (Algebra.botEquiv F E)


theorem botEquiv_def (x : F) : botEquiv F E (algebraMap F (⊥ : IntermediateField F E) x) = x := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x : F
    ⊢ Eq ((IntermediateField.botEquiv F E) ((algebraMap F (Subtype fun x => Member …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem botEquiv_symm (x : F) : (botEquiv F E).symm x = algebraMap F _ x :=
  rfl


noncomputable instance algebraOverBot : Algebra (⊥ : IntermediateField F E) F :=
  (IntermediateField.botEquiv F E).toAlgHom.toRingHom.toAlgebra


theorem coe_algebraMap_over_bot :
    (algebraMap (⊥ : IntermediateField F E) F : (⊥ : IntermediateField F E) → F) =
      IntermediateField.botEquiv F E :=
  rfl


instance isScalarTower_over_bot : IsScalarTower (⊥ : IntermediateField F E) F E :=
  IsScalarTower.of_algebraMap_eq
    (by
      /-
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        ⊢ ∀ (x : Subtype fun x => Membership.mem Bot.bot x), Eq ((algebraMap (Subtype  …
      -/
      intro x
      /-
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        x : Subtype fun x => Membership.mem Bot.bot x
        ⊢ Eq ((algebraMap (Subtype fun x => Membership.mem Bot.bot x) E) x) ((algebraM …
      -/
      obtain ⟨y, rfl⟩ := (botEquiv F E).symm.surjective x
      rw [coe_algebraMap_over_bot, (botEquiv F E).apply_symm_apply, botEquiv_symm,
        IsScalarTower.algebraMap_apply F (⊥ : IntermediateField F E) E])


/-- The top `IntermediateField` is isomorphic to the field.

This is the intermediate field version of `Subalgebra.topEquiv`. -/
@[simps!]
def topEquiv : (⊤ : IntermediateField F E) ≃ₐ[F] E :=
  Subalgebra.topEquiv

-- Porting note: this theorem is now generated by the `@[simps!]` above.


@[simp]
theorem restrictScalars_bot_eq_self (K : IntermediateField F E) :
    (⊥ : IntermediateField K E).restrictScalars _ = K :=
  SetLike.coe_injective Subtype.range_coe


@[simp]
theorem restrictScalars_top : (⊤ : IntermediateField F E).restrictScalars K = ⊤ :=
  rfl


theorem restrictScalars_sup :
    L.restrictScalars K ⊔ L'.restrictScalars K = (L ⊔ L').restrictScalars K :=
                           /-
                             F : Type u_1
                             inst✝⁶ : Field F
                             E : Type u_2
                             inst✝⁵ : Field E
                             inst✝⁴ : Algebra F E
                             K : Type u_3
                             inst✝³ : Field K
                             inst✝² : Algebra K E
                             inst✝¹ : Algebra K F
                             inst✝ : IsScalarTower K F E
                             L L' : IntermediateField F E
                             ⊢ Eq (Max.max (IntermediateField.restrictScalars K L) (IntermediateField.restr …
                           -/
  toSubfield_injective (by simp)
                           /-
                             🎉 no goals
                           -/


theorem restrictScalars_inf :
    L.restrictScalars K ⊓ L'.restrictScalars K = (L ⊓ L').restrictScalars K := rfl


@[simp]
theorem map_bot (f : E →ₐ[F] K) :
    IntermediateField.map f ⊥ = ⊥ :=
  toSubalgebra_injective <| Algebra.map_bot _


theorem map_sup (s t : IntermediateField F E) (f : E →ₐ[F] K) : (s ⊔ t).map f = s.map f ⊔ t.map f :=
  (gc_map_comap f).l_sup


theorem map_iSup {ι : Sort*} (f : E →ₐ[F] K) (s : ι → IntermediateField F E) :
    (iSup s).map f = ⨆ i, (s i).map f :=
  (gc_map_comap f).l_iSup


theorem map_inf (s t : IntermediateField F E) (f : E →ₐ[F] K) :
    (s ⊓ t).map f = s.map f ⊓ t.map f := SetLike.coe_injective (Set.image_inter f.injective)


theorem map_iInf {ι : Sort*} [Nonempty ι] (f : E →ₐ[F] K) (s : ι → IntermediateField F E) :
    (iInf s).map f = ⨅ i, (s i).map f := by
  /-
    F : Type u_1
    inst✝⁵ : Field F
    E : Type u_2
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    K : Type u_3
    inst✝² : Field K
    inst✝¹ : Algebra F K
    ι : Sort u_4
    inst✝ : Nonempty ι
    f : AlgHom F E K
    s : ι → IntermediateField F E
    ⊢ Eq (IntermediateField.map f (iInf s)) (iInf fun i => IntermediateField.map f …
  -/
  apply SetLike.coe_injective
  /-
    case a
    F : Type u_1
    inst✝⁵ : Field F
    E : Type u_2
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    K : Type u_3
    inst✝² : Field K
    inst✝¹ : Algebra F K
    ι : Sort u_4
    inst✝ : Nonempty ι
    f : AlgHom F E K
    s : ι → IntermediateField F E
    ⊢ Eq ↑(IntermediateField.map f (iInf s)) ↑(iInf fun i => IntermediateField.map …
  -/
  simpa using (Set.injOn_of_injective f.injective).image_iInter_eq (s := SetLike.coe ∘ s)
  /-
    🎉 no goals
  -/


theorem _root_.AlgHom.fieldRange_eq_map (f : E →ₐ[F] K) :
    f.fieldRange = IntermediateField.map f ⊤ :=
  SetLike.ext' Set.image_univ.symm


theorem _root_.AlgHom.map_fieldRange {L : Type*} [Field L] [Algebra F L]
    (f : E →ₐ[F] K) (g : K →ₐ[F] L) : f.fieldRange.map g = (g.comp f).fieldRange :=
  SetLike.ext' (Set.range_comp g f).symm


theorem _root_.AlgHom.fieldRange_eq_top {f : E →ₐ[F] K} :
    f.fieldRange = ⊤ ↔ Function.Surjective f :=
  SetLike.ext'_iff.trans Set.range_eq_univ


@[simp]
theorem _root_.AlgEquiv.fieldRange_eq_top (f : E ≃ₐ[F] K) :
    (f : E →ₐ[F] K).fieldRange = ⊤ :=
  AlgHom.fieldRange_eq_top.mpr f.surjective


theorem fieldRange_comp_val : (f.comp L.val).fieldRange = L.map f := toSubalgebra_injective <| by
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type u_3
    inst✝¹ : Field K
    inst✝ : Algebra F K
    L : IntermediateField F E
    f : AlgHom F E K
    ⊢ Eq (f.comp L.val).fieldRange.toSubalgebra (IntermediateField.map f L).toSuba …
  -/
  rw [toSubalgebra_map, AlgHom.fieldRange_toSubalgebra, AlgHom.range_comp, range_val]
  /-
    🎉 no goals
  -/


/-- An intermediate field is isomorphic to its image under an `AlgHom`
(which is automatically injective) -/
noncomputable def equivMap : L ≃ₐ[F] L.map f :=
  (AlgEquiv.ofInjective _ (f.comp L.val).injective).trans (equivOfEq (fieldRange_comp_val L f))


@[simp]
theorem coe_equivMap_apply (x : L) : ↑(equivMap L f x) = f x := rfl


theorem adjoin_eq_range_algebraMap_adjoin :
    (adjoin F S : Set E) = Set.range (algebraMap (adjoin F S) E) :=
  Subtype.range_coe.symm


theorem adjoin.algebraMap_mem (x : F) : algebraMap F E x ∈ adjoin F S :=
  IntermediateField.algebraMap_mem (adjoin F S) x


theorem adjoin.range_algebraMap_subset : Set.range (algebraMap F E) ⊆ adjoin F S := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    ⊢ HasSubset.Subset (Set.range ⇑(algebraMap F E)) ↑(IntermediateField.adjoin F S)
  -/
  intro x hx
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    x : E
    hx : Membership.mem (Set.range ⇑(algebraMap F E)) x
    ⊢ Membership.mem (↑(IntermediateField.adjoin F S)) x
  -/
  cases' hx with f hf
  /-
    case intro
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    x : E
    f : F
    hf : Eq ((algebraMap F E) f) x
    ⊢ Membership.mem (↑(IntermediateField.adjoin F S)) x
  -/
  rw [← hf]
  /-
    case intro
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    x : E
    f : F
    hf : Eq ((algebraMap F E) f) x
    ⊢ Membership.mem (↑(IntermediateField.adjoin F S)) ((algebraMap F E) f)
  -/
  exact adjoin.algebraMap_mem F S f
  /-
    🎉 no goals
  -/


instance adjoin.fieldCoe : CoeTC F (adjoin F S) where
  coe x := ⟨algebraMap F E x, adjoin.algebraMap_mem F S x⟩


theorem subset_adjoin : S ⊆ adjoin F S := fun _ hx => Subfield.subset_closure (Or.inr hx)


instance adjoin.setCoe : CoeTC S (adjoin F S) where coe x := ⟨x, subset_adjoin F S (Subtype.mem x)⟩


@[mono]
theorem adjoin.mono (T : Set E) (h : S ⊆ T) : adjoin F S ≤ adjoin F T :=
  GaloisConnection.monotone_l gc h


theorem adjoin_contains_field_as_subfield (F : Subfield E) : (F : Set E) ⊆ adjoin F S := fun x hx =>
  adjoin.algebraMap_mem F S ⟨x, hx⟩


theorem subset_adjoin_of_subset_left {F : Subfield E} {T : Set E} (HT : T ⊆ F) : T ⊆ adjoin F S :=
  fun x hx => (adjoin F S).algebraMap_mem ⟨x, HT hx⟩


theorem subset_adjoin_of_subset_right {T : Set E} (H : T ⊆ S) : T ⊆ adjoin F S := fun _ hx =>
  subset_adjoin F S (H hx)


@[simp]
theorem adjoin_empty (F E : Type*) [Field F] [Field E] [Algebra F E] : adjoin F (∅ : Set E) = ⊥ :=
  eq_bot_iff.mpr (adjoin_le_iff.mpr (Set.empty_subset _))


@[simp]
theorem adjoin_univ (F E : Type*) [Field F] [Field E] [Algebra F E] :
    adjoin F (Set.univ : Set E) = ⊤ :=
  eq_top_iff.mpr <| subset_adjoin _ _


/-- If `K` is a field with `F ⊆ K` and `S ⊆ K` then `adjoin F S ≤ K`. -/
theorem adjoin_le_subfield {K : Subfield E} (HF : Set.range (algebraMap F E) ⊆ K) (HS : S ⊆ K) :
    (adjoin F S).toSubfield ≤ K := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    K : Subfield E
    HF : HasSubset.Subset (Set.range ⇑(algebraMap F E)) ↑K
    HS : HasSubset.Subset S ↑K
    ⊢ LE.le (IntermediateField.adjoin F S).toSubfield K
  -/
  apply Subfield.closure_le.mpr
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    K : Subfield E
    HF : HasSubset.Subset (Set.range ⇑(algebraMap F E)) ↑K
    HS : HasSubset.Subset S ↑K
    ⊢ HasSubset.Subset (Union.union (Set.range ⇑(algebraMap F E)) S) ↑K
  -/
  rw [Set.union_subset_iff]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    K : Subfield E
    HF : HasSubset.Subset (Set.range ⇑(algebraMap F E)) ↑K
    HS : HasSubset.Subset S ↑K
    ⊢ And (HasSubset.Subset (Set.range ⇑(algebraMap F E)) ↑K) (HasSubset.Subset S  …
  -/
  exact ⟨HF, HS⟩
  /-
    🎉 no goals
  -/


theorem adjoin_subset_adjoin_iff {F' : Type*} [Field F'] [Algebra F' E] {S S' : Set E} :
    (adjoin F S : Set E) ⊆ adjoin F' S' ↔
      Set.range (algebraMap F E) ⊆ adjoin F' S' ∧ S ⊆ adjoin F' S' :=
  ⟨fun h => ⟨(adjoin.range_algebraMap_subset _ _).trans h,
    (subset_adjoin _ _).trans h⟩, fun ⟨hF, hS⟩ =>
      (Subfield.closure_le (t := (adjoin F' S').toSubfield)).mpr (Set.union_subset hF hS)⟩


/-- `F[S][T] = F[S ∪ T]` -/
theorem adjoin_adjoin_left (T : Set E) :
    (adjoin (adjoin F S) T).restrictScalars _ = adjoin F (S ∪ T) := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S T : Set E
    ⊢ Eq (IntermediateField.restrictScalars F (IntermediateField.adjoin (Subtype f …
  -/
  rw [SetLike.ext'_iff]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S T : Set E
    ⊢ Eq ↑(IntermediateField.restrictScalars F (IntermediateField.adjoin (Subtype  …
  -/
  change (adjoin (adjoin F S) T : Set E) = _
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S T : Set E
    ⊢ Eq ↑(IntermediateField.adjoin (Subtype fun x => Membership.mem (Intermediate …
  -/
  apply subset_antisymm <;> rw [adjoin_subset_adjoin_iff] <;> constructor
    /-
      case a.left
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      S T : Set E
      ⊢ HasSubset.Subset (Set.range ⇑(algebraMap (Subtype fun x => Membership.mem (I …
    -/
  · rintro _ ⟨⟨x, hx⟩, rfl⟩; exact adjoin.mono _ _ _ Set.subset_union_left hx
                             /-
                               🎉 no goals
                             -/
    /-
      case a.right
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      S T : Set E
      ⊢ HasSubset.Subset T ↑(IntermediateField.adjoin F (Union.union S T))
    -/
  · exact subset_adjoin_of_subset_right _ _ Set.subset_union_right
    /-
      🎉 no goals
    -/
    /-
      case a.left
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      S T : Set E
      ⊢ HasSubset.Subset (Set.range ⇑(algebraMap F E)) ↑(IntermediateField.adjoin (S …
    -/
  · exact Set.range_subset_iff.mpr fun f ↦ Subfield.subset_closure (.inl ⟨f, rfl⟩)
    /-
      🎉 no goals
    -/
  · exact Set.union_subset
      (fun x hx ↦ Subfield.subset_closure <| .inl ⟨⟨x, Subfield.subset_closure (.inr hx)⟩, rfl⟩)
      (fun x hx ↦ Subfield.subset_closure <| .inr hx)


@[simp]
theorem adjoin_insert_adjoin (x : E) :
    adjoin F (insert x (adjoin F S : Set E)) = adjoin F (insert x S) :=
  le_antisymm
    (adjoin_le_iff.mpr
      (Set.insert_subset_iff.mpr
        ⟨subset_adjoin _ _ (Set.mem_insert _ _),
          adjoin_le_iff.mpr (subset_adjoin_of_subset_right _ _ (Set.subset_insert _ _))⟩))
    (adjoin.mono _ _ _ (Set.insert_subset_insert (subset_adjoin _ _)))


/-- `F[S][T] = F[T][S]` -/
theorem adjoin_adjoin_comm (T : Set E) :
    (adjoin (adjoin F S) T).restrictScalars F = (adjoin (adjoin F T) S).restrictScalars F := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S T : Set E
    ⊢ Eq (IntermediateField.restrictScalars F (IntermediateField.adjoin (Subtype f …
  -/
  rw [adjoin_adjoin_left, adjoin_adjoin_left, Set.union_comm]
  /-
    🎉 no goals
  -/


theorem adjoin_map {E' : Type*} [Field E'] [Algebra F E'] (f : E →ₐ[F] E') :
    (adjoin F S).map f = adjoin F (f '' S) :=
  le_antisymm
    (map_le_iff_le_comap.mpr <| adjoin_le_iff.mpr fun x hx ↦ subset_adjoin _ _ ⟨x, hx, rfl⟩)
    (adjoin_le_iff.mpr <| Set.monotone_image <| subset_adjoin _ _)


@[simp]
theorem lift_adjoin (K : IntermediateField F E) (S : Set K) :
    lift (adjoin F S) = adjoin F (Subtype.val '' S) :=
  adjoin_map _ _ _


theorem lift_adjoin_simple (K : IntermediateField F E) (α : K) :
    lift (adjoin F {α}) = adjoin F {α.1} := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    K : IntermediateField F E
    α : Subtype fun x => Membership.mem K x
    ⊢ Eq (IntermediateField.lift (IntermediateField.adjoin F (Singleton.singleton  …
  -/
  simp only [lift_adjoin, Set.image_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_bot (K : IntermediateField F E) :
    lift (F := K) ⊥ = ⊥ := map_bot _


@[simp]
theorem lift_top (K : IntermediateField F E) :
                              /-
                                F : Type u_1
                                inst✝² : Field F
                                E : Type u_2
                                inst✝¹ : Field E
                                inst✝ : Algebra F E
                                K : IntermediateField F E
                                ⊢ Eq (IntermediateField.lift Top.top) K
                              -/
    lift (F := K) ⊤ = K := by rw [lift, ← AlgHom.fieldRange_eq_map, fieldRange_val]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem adjoin_self (K : IntermediateField F E) :
    adjoin F K = K := le_antisymm (adjoin_le_iff.2 fun _ ↦ id) (subset_adjoin F _)


theorem restrictScalars_adjoin (K : IntermediateField F E) (S : Set E) :
    restrictScalars F (adjoin K S) = adjoin F (K ∪ S) := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    K : IntermediateField F E
    S : Set E
    ⊢ Eq (IntermediateField.restrictScalars F (IntermediateField.adjoin (Subtype f …
  -/
  rw [← adjoin_self _ K, adjoin_adjoin_left, adjoin_self _ K]
  /-
    🎉 no goals
  -/


variable {F} in
theorem extendScalars_adjoin {K : IntermediateField F E} {S : Set E} (h : K ≤ adjoin F S) :
    extendScalars h = adjoin K S := restrictScalars_injective F <| by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    K : IntermediateField F E
    S : Set E
    h : LE.le K (IntermediateField.adjoin F S)
    ⊢ Eq (IntermediateField.restrictScalars F (IntermediateField.extendScalars h)) …
  -/
  rw [extendScalars_restrictScalars, restrictScalars_adjoin]
  exact le_antisymm (adjoin.mono F S _ Set.subset_union_right) <| adjoin_le_iff.2 <|
    Set.union_subset h (subset_adjoin F S)


theorem adjoin_union {S T : Set E} : adjoin F (S ∪ T) = adjoin F S ⊔ adjoin F T :=
  gc.l_sup


theorem restrictScalars_adjoin_eq_sup (K : IntermediateField F E) (S : Set E) :
    restrictScalars F (adjoin K S) = K ⊔ adjoin F S := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    K : IntermediateField F E
    S : Set E
    ⊢ Eq (IntermediateField.restrictScalars F (IntermediateField.adjoin (Subtype f …
  -/
  rw [restrictScalars_adjoin, adjoin_union, adjoin_self]
  /-
    🎉 no goals
  -/


theorem adjoin_iUnion {ι} (f : ι → Set E) : adjoin F (⋃ i, f i) = ⨆ i, adjoin F (f i) :=
  gc.l_iSup


theorem iSup_eq_adjoin {ι} (f : ι → IntermediateField F E) :
    ⨆ i, f i = adjoin F (⋃ i, f i : Set E) := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ι : Sort u_3
    f : ι → IntermediateField F E
    ⊢ Eq (iSup fun i => f i) (IntermediateField.adjoin F (Set.iUnion fun i => ↑(f  …
  -/
  simp_rw [adjoin_iUnion, adjoin_self]
  /-
    🎉 no goals
  -/


variable {F} in
/-- If `E / L / F` and `E / L' / F` are two field extension towers, `L ≃ₐ[F] L'` is an isomorphism
compatible with `E / L` and `E / L'`, then for any subset `S` of `E`, `L(S)` and `L'(S)` are
equal as intermediate fields of `E / F`. -/
theorem restrictScalars_adjoin_of_algEquiv
    {L L' : Type*} [Field L] [Field L']
    [Algebra F L] [Algebra L E] [Algebra F L'] [Algebra L' E]
    [IsScalarTower F L E] [IsScalarTower F L' E] (i : L ≃ₐ[F] L')
    (hi : algebraMap L E = (algebraMap L' E) ∘ i) (S : Set E) :
    (adjoin L S).restrictScalars F = (adjoin L' S).restrictScalars F := by
  apply_fun toSubfield using (fun K K' h ↦ by
    ext x; change x ∈ K.toSubfield ↔ x ∈ K'.toSubfield; rw [h])
  /-
    F : Type u_1
    inst✝¹⁰ : Field F
    E : Type u_2
    inst✝⁹ : Field E
    inst✝⁸ : Algebra F E
    L : Type u_3
    L' : Type u_4
    inst✝⁷ : Field L
    inst✝⁶ : Field L'
    inst✝⁵ : Algebra F L
    inst✝⁴ : Algebra L E
    inst✝³ : Algebra F L'
    inst✝² : Algebra L' E
    inst✝¹ : IsScalarTower F L E
    inst✝ : IsScalarTower F L' E
    i : AlgEquiv F L L'
    hi : Eq (⇑(algebraMap L E)) (Function.comp ⇑(algebraMap L' E) ⇑i)
    S : Set E
    ⊢ Eq (IntermediateField.restrictScalars F (IntermediateField.adjoin L S)).toSu …
  -/
  change Subfield.closure _ = Subfield.closure _
  /-
    F : Type u_1
    inst✝¹⁰ : Field F
    E : Type u_2
    inst✝⁹ : Field E
    inst✝⁸ : Algebra F E
    L : Type u_3
    L' : Type u_4
    inst✝⁷ : Field L
    inst✝⁶ : Field L'
    inst✝⁵ : Algebra F L
    inst✝⁴ : Algebra L E
    inst✝³ : Algebra F L'
    inst✝² : Algebra L' E
    inst✝¹ : IsScalarTower F L E
    inst✝ : IsScalarTower F L' E
    i : AlgEquiv F L L'
    hi : Eq (⇑(algebraMap L E)) (Function.comp ⇑(algebraMap L' E) ⇑i)
    S : Set E
    ⊢ Eq (Subfield.closure (Union.union (Set.range ⇑(algebraMap L E)) S)) (Subfiel …
  -/
  congr
  /-
    case e_s.e_a
    F : Type u_1
    inst✝¹⁰ : Field F
    E : Type u_2
    inst✝⁹ : Field E
    inst✝⁸ : Algebra F E
    L : Type u_3
    L' : Type u_4
    inst✝⁷ : Field L
    inst✝⁶ : Field L'
    inst✝⁵ : Algebra F L
    inst✝⁴ : Algebra L E
    inst✝³ : Algebra F L'
    inst✝² : Algebra L' E
    inst✝¹ : IsScalarTower F L E
    inst✝ : IsScalarTower F L' E
    i : AlgEquiv F L L'
    hi : Eq (⇑(algebraMap L E)) (Function.comp ⇑(algebraMap L' E) ⇑i)
    S : Set E
    ⊢ Eq (Set.range ⇑(algebraMap L E)) (Set.range ⇑(algebraMap L' E))
  -/
  ext x
  exact ⟨fun ⟨y, h⟩ ↦ ⟨i y, by rw [← h, hi]; rfl⟩,
    fun ⟨y, h⟩ ↦ ⟨i.symm y, by rw [← h, hi, Function.comp_apply, AlgEquiv.apply_symm_apply]⟩⟩


theorem algebra_adjoin_le_adjoin : Algebra.adjoin F S ≤ (adjoin F S).toSubalgebra :=
  Algebra.adjoin_le (subset_adjoin _ _)


theorem adjoin_eq_algebra_adjoin (inv_mem : ∀ x ∈ Algebra.adjoin F S, x⁻¹ ∈ Algebra.adjoin F S) :
    (adjoin F S).toSubalgebra = Algebra.adjoin F S :=
  le_antisymm
    (show adjoin F S ≤
        { Algebra.adjoin F S with
          inv_mem' := inv_mem }
      from adjoin_le_iff.mpr Algebra.subset_adjoin)
    (algebra_adjoin_le_adjoin _ _)


theorem eq_adjoin_of_eq_algebra_adjoin (K : IntermediateField F E)
    (h : K.toSubalgebra = Algebra.adjoin F S) : K = adjoin F S := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    K : IntermediateField F E
    h : Eq K.toSubalgebra (Algebra.adjoin F S)
    ⊢ Eq K (IntermediateField.adjoin F S)
  -/
  apply toSubalgebra_injective
  /-
    case a
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    K : IntermediateField F E
    h : Eq K.toSubalgebra (Algebra.adjoin F S)
    ⊢ Eq K.toSubalgebra (IntermediateField.adjoin F S).toSubalgebra
  -/
  rw [h]
  /-
    case a
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    K : IntermediateField F E
    h : Eq K.toSubalgebra (Algebra.adjoin F S)
    ⊢ Eq (Algebra.adjoin F S) (IntermediateField.adjoin F S).toSubalgebra
  -/
  refine (adjoin_eq_algebra_adjoin F _ fun x ↦ ?_).symm
  /-
    case a
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    K : IntermediateField F E
    h : Eq K.toSubalgebra (Algebra.adjoin F S)
    x : E
    ⊢ Membership.mem (Algebra.adjoin F S) x → Membership.mem (Algebra.adjoin F S)  …
  -/
  rw [← h]
  /-
    case a
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    K : IntermediateField F E
    h : Eq K.toSubalgebra (Algebra.adjoin F S)
    x : E
    ⊢ Membership.mem K.toSubalgebra x → Membership.mem K.toSubalgebra (Inv.inv x)
  -/
  exact K.inv_mem
  /-
    🎉 no goals
  -/


theorem adjoin_eq_top_of_algebra (hS : Algebra.adjoin F S = ⊤) : adjoin F S = ⊤ :=
  top_le_iff.mp (hS.symm.trans_le <| algebra_adjoin_le_adjoin F S)


@[elab_as_elim]
theorem adjoin_induction {s : Set E} {p : ∀ x ∈ adjoin F s, Prop}
    (mem : ∀ x hx, p x (subset_adjoin _ _ hx))
    (algebraMap : ∀ x, p (algebraMap F E x) (algebraMap_mem _ _))
    (add : ∀ x y hx hy, p x hx → p y hy → p (x + y) (add_mem hx hy))
    (inv : ∀ x hx, p x hx → p x⁻¹ (inv_mem hx))
    (mul : ∀ x y hx hy, p x hx → p y hy → p (x * y) (mul_mem hx hy))
    {x} (h : x ∈ adjoin F s) : p x h :=
  Subfield.closure_induction
    (fun x hx ↦ Or.casesOn hx (fun ⟨x, hx⟩ ↦ hx ▸ algebraMap x) (mem x))
        /-
          F : Type u_1
          inst✝² : Field F
          E : Type u_2
          inst✝¹ : Field E
          inst✝ : Algebra F E
          s : Set E
          p : (x : E) → Membership.mem (IntermediateField.adjoin F s) x → Prop
          mem : ∀ (x : E) (hx : Membership.mem s x), p x ⋯
          algebraMap : ∀ (x : F), p ((_root_.algebraMap F E) x) ⋯
          add : ∀ (x y : E) (hx : Membership.mem (IntermediateField.adjoin F s) x) (hy : …
          inv : ∀ (x : E) (hx : Membership.mem (IntermediateField.adjoin F s) x), p x hx …
          mul : ∀ (x y : E) (hx : Membership.mem (IntermediateField.adjoin F s) x) (hy : …
          x : E
          h : Membership.mem (IntermediateField.adjoin F s) x
          ⊢ p 1 ⋯
        -/
    (by simp_rw [← (_root_.algebraMap F E).map_one]; exact algebraMap 1) add
                                                     /-
                                                       🎉 no goals
                                                     -/
    (fun x _ h ↦ by
      /-
        F : Type u_1
        inst✝² : Field F
        E : Type u_2
        inst✝¹ : Field E
        inst✝ : Algebra F E
        s : Set E
        p : (x : E) → Membership.mem (IntermediateField.adjoin F s) x → Prop
        mem : ∀ (x : E) (hx : Membership.mem s x), p x ⋯
        algebraMap : ∀ (x : F), p ((_root_.algebraMap F E) x) ⋯
        add : ∀ (x y : E) (hx : Membership.mem (IntermediateField.adjoin F s) x) (hy : …
        inv : ∀ (x : E) (hx : Membership.mem (IntermediateField.adjoin F s) x), p x hx …
        mul : ∀ (x y : E) (hx : Membership.mem (IntermediateField.adjoin F s) x) (hy : …
        x✝¹ : E
        h✝ : Membership.mem (IntermediateField.adjoin F s) x✝¹
        x : E
        x✝ : Membership.mem (Subfield.closure (Union.union (Set.range ⇑(_root_.algebra …
        h : p x x✝
        ⊢ p (Neg.neg x) ⋯
      -/
      simp_rw [← neg_one_smul F x, Algebra.smul_def]; exact mul _ _ _ _ (algebraMap _) h) inv mul h
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem adjoin_algHom_ext {s : Set E} ⦃φ₁ φ₂ : adjoin F s →ₐ[F] K⦄
    (h : ∀ x hx, φ₁ ⟨x, subset_adjoin _ _ hx⟩ = φ₂ ⟨x, subset_adjoin _ _ hx⟩) :
    φ₁ = φ₂ :=
  AlgHom.ext fun ⟨x, hx⟩ ↦ adjoin_induction _ h (fun _ ↦ φ₂.commutes _ ▸ φ₁.commutes _)
                            /-
                              F : Type u_1
                              inst✝⁴ : Field F
                              E : Type u_2
                              inst✝³ : Field E
                              inst✝² : Algebra F E
                              K : Type u_3
                              inst✝¹ : Semiring K
                              inst✝ : Algebra F K
                              s : Set E
                              φ₁ φ₂ : AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F  …
                              h : ∀ (x : E) (hx : Membership.mem s x), Eq (φ₁ ⟨x, ⋯⟩) (φ₂ ⟨x, ⋯⟩)
                              x✝⁴ : Subtype fun x => Membership.mem (IntermediateField.adjoin F s) x
                              x : E
                              hx : Membership.mem (IntermediateField.adjoin F s) x
                              x✝³ x✝² : E
                              x✝¹ : Membership.mem (IntermediateField.adjoin F s) x✝³
                              x✝ : Membership.mem (IntermediateField.adjoin F s) x✝²
                              h₁ : Eq (φ₁ ⟨x✝³, x✝¹⟩) (φ₂ ⟨x✝³, x✝¹⟩)
                              h₂ : Eq (φ₁ ⟨x✝², x✝⟩) (φ₂ ⟨x✝², x✝⟩)
                              ⊢ Eq (φ₁ ⟨HAdd.hAdd x✝³ x✝², ⋯⟩) (φ₂ ⟨HAdd.hAdd x✝³ x✝², ⋯⟩)
                            -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    (fun _ _ _ _ h₁ h₂ ↦ by convert congr_arg₂ (· + ·) h₁ h₂ <;> rw [← map_add] <;> rfl)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    (fun _ _ ↦ eq_on_inv₀ _ _)
                            /-
                              F : Type u_1
                              inst✝⁴ : Field F
                              E : Type u_2
                              inst✝³ : Field E
                              inst✝² : Algebra F E
                              K : Type u_3
                              inst✝¹ : Semiring K
                              inst✝ : Algebra F K
                              s : Set E
                              φ₁ φ₂ : AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F  …
                              h : ∀ (x : E) (hx : Membership.mem s x), Eq (φ₁ ⟨x, ⋯⟩) (φ₂ ⟨x, ⋯⟩)
                              x✝⁴ : Subtype fun x => Membership.mem (IntermediateField.adjoin F s) x
                              x : E
                              hx : Membership.mem (IntermediateField.adjoin F s) x
                              x✝³ x✝² : E
                              x✝¹ : Membership.mem (IntermediateField.adjoin F s) x✝³
                              x✝ : Membership.mem (IntermediateField.adjoin F s) x✝²
                              h₁ : Eq (φ₁ ⟨x✝³, x✝¹⟩) (φ₂ ⟨x✝³, x✝¹⟩)
                              h₂ : Eq (φ₁ ⟨x✝², x✝⟩) (φ₂ ⟨x✝², x✝⟩)
                              ⊢ Eq (φ₁ ⟨HMul.hMul x✝³ x✝², ⋯⟩) (φ₂ ⟨HMul.hMul x✝³ x✝², ⋯⟩)
                            -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    (fun _ _ _ _ h₁ h₂ ↦ by convert congr_arg₂ (· * ·) h₁ h₂ <;> rw [← map_mul] <;> rfl)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    hx


theorem algHom_ext_of_eq_adjoin {S : IntermediateField F E} {s : Set E} (hS : S = adjoin F s)
    ⦃φ₁ φ₂ : S →ₐ[F] K⦄
    (h : ∀ x hx, φ₁ ⟨x, hS.ge (subset_adjoin _ _ hx)⟩ = φ₂ ⟨x, hS.ge (subset_adjoin _ _ hx)⟩) :
    φ₁ = φ₂ := by
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type u_3
    inst✝¹ : Semiring K
    inst✝ : Algebra F K
    S : IntermediateField F E
    s : Set E
    hS : Eq S (IntermediateField.adjoin F s)
    φ₁ φ₂ : AlgHom F (Subtype fun x => Membership.mem S x) K
    h : ∀ (x : E) (hx : Membership.mem s x), Eq (φ₁ ⟨x, ⋯⟩) (φ₂ ⟨x, ⋯⟩)
    ⊢ Eq φ₁ φ₂
  -/
  subst hS; exact adjoin_algHom_ext F h
            /-
              🎉 no goals
            -/


open Lean in
/-- Supporting function for the `F⟮x₁,x₂,...,xₙ⟯` adjunction notation. -/
private partial def mkInsertTerm {m : Type → Type} [Monad m] [MonadQuotation m]
    (xs : TSyntaxArray `term) : m Term := run 0 where
  run (i : Nat) : m Term := do
    if i + 1 == xs.size then
      ``(singleton $(xs[i]!))
    else if i < xs.size then
      ``(insert $(xs[i]!) $(← run (i + 1)))
    else
      ``(EmptyCollection.emptyCollection)


/-- If `x₁ x₂ ... xₙ : E` then `F⟮x₁,x₂,...,xₙ⟯` is the `IntermediateField F E`
generated by these elements. -/
scoped macro:max K:term "⟮" xs:term,* "⟯" : term => do ``(adjoin $K $(← mkInsertTerm xs.getElems))


open Lean PrettyPrinter.Delaborator SubExpr in
@[app_delab IntermediateField.adjoin]
partial def delabAdjoinNotation : Delab := whenPPOption getPPNotation do
  let e ← getExpr
  guard <| e.isAppOfArity ``adjoin 6
  let F ← withNaryArg 0 delab
  let xs ← withNaryArg 5 delabInsertArray
  `($F⟮$(xs.toArray),*⟯)
where
  delabInsertArray : DelabM (List Term) := do
    let e ← getExpr
    if e.isAppOfArity ``EmptyCollection.emptyCollection 2 then
      return []
    else if e.isAppOfArity ``singleton 4 then
      let x ← withNaryArg 3 delab
      return [x]
    else if e.isAppOfArity ``insert 5 then
      let x ← withNaryArg 3 delab
      let xs ← withNaryArg 4 delabInsertArray
      return x :: xs
    else failure


theorem mem_adjoin_simple_self : α ∈ F⟮α⟯ :=
  subset_adjoin F {α} (Set.mem_singleton α)


/-- generator of `F⟮α⟯` -/
def AdjoinSimple.gen : F⟮α⟯ :=
  ⟨α, mem_adjoin_simple_self F α⟩


@[simp]
theorem AdjoinSimple.coe_gen : (AdjoinSimple.gen F α : E) = α :=
  rfl


theorem AdjoinSimple.algebraMap_gen : algebraMap F⟮α⟯ E (AdjoinSimple.gen F α) = α :=
  rfl


@[simp]
theorem AdjoinSimple.isIntegral_gen : IsIntegral F (AdjoinSimple.gen F α) ↔ IsIntegral F α := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    ⊢ Iff (IsIntegral F (IntermediateField.AdjoinSimple.gen F α)) (IsIntegral F α)
  -/
  conv_rhs => rw [← AdjoinSimple.algebraMap_gen F α]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    ⊢ Iff (IsIntegral F (IntermediateField.AdjoinSimple.gen F α)) (IsIntegral F (( …
  -/
  rw [isIntegral_algebraMap_iff (algebraMap F⟮α⟯ E).injective]
  /-
    🎉 no goals
  -/


theorem adjoin_simple_adjoin_simple (β : E) : F⟮α⟯⟮β⟯.restrictScalars F = F⟮α, β⟯ :=
  adjoin_adjoin_left _ _ _


theorem adjoin_simple_comm (β : E) : F⟮α⟯⟮β⟯.restrictScalars F = F⟮β⟯⟮α⟯.restrictScalars F :=
  adjoin_adjoin_comm _ _ _


theorem adjoin_algebraic_toSubalgebra {S : Set E} (hS : ∀ x ∈ S, IsAlgebraic F x) :
    (IntermediateField.adjoin F S).toSubalgebra = Algebra.adjoin F S :=
  adjoin_eq_algebra_adjoin _ _ fun _ ↦
    (Algebra.IsIntegral.adjoin fun x hx ↦ (hS x hx).isIntegral).inv_mem


theorem adjoin_simple_toSubalgebra_of_integral (hα : IsIntegral F α) :
    F⟮α⟯.toSubalgebra = Algebra.adjoin F {α} := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    hα : IsIntegral F α
    ⊢ Eq (IntermediateField.adjoin F (Singleton.singleton α)).toSubalgebra (Algebr …
  -/
  apply adjoin_algebraic_toSubalgebra
  /-
    case hS
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    hα : IsIntegral F α
    ⊢ ∀ (x : E), Membership.mem (Singleton.singleton α) x → IsAlgebraic F x
  -/
  rintro x (rfl : x = α)
  /-
    case hS
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x : E
    hα : IsIntegral F x
    ⊢ IsAlgebraic F x
  -/
  rwa [isAlgebraic_iff_isIntegral]
  /-
    🎉 no goals
  -/


/-- Characterize `IsSplittingField` with `IntermediateField.adjoin` instead of `Algebra.adjoin`. -/
theorem _root_.isSplittingField_iff_intermediateField {p : F[X]} :
    p.IsSplittingField F E ↔ p.Splits (algebraMap F E) ∧ adjoin F (p.rootSet E) = ⊤ := by
  rw [← toSubalgebra_injective.eq_iff,
      adjoin_algebraic_toSubalgebra fun _ ↦ isAlgebraic_of_mem_rootSet]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    ⊢ Iff (Polynomial.IsSplittingField F E p) (And (Polynomial.Splits (algebraMap  …
  -/
  exact ⟨fun ⟨spl, adj⟩ ↦ ⟨spl, adj⟩, fun ⟨spl, adj⟩ ↦ ⟨spl, adj⟩⟩
  /-
    🎉 no goals
  -/

-- Note: p.Splits (algebraMap F E) also works

theorem isSplittingField_iff {p : F[X]} {K : IntermediateField F E} :
    p.IsSplittingField F K ↔ p.Splits (algebraMap F K) ∧ K = adjoin F (p.rootSet E) := by
  suffices _ → (Algebra.adjoin F (p.rootSet K) = ⊤ ↔ K = adjoin F (p.rootSet E)) by
    exact ⟨fun h ↦ ⟨h.1, (this h.1).mp h.2⟩, fun h ↦ ⟨h.1, (this h.1).mpr h.2⟩⟩
  rw [← toSubalgebra_injective.eq_iff,
      adjoin_algebraic_toSubalgebra fun x ↦ isAlgebraic_of_mem_rootSet]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    K : IntermediateField F E
    ⊢ Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem K x)) p → I …
  -/
  refine fun hp ↦ (adjoin_rootSet_eq_range hp K.val).symm.trans ?_
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    K : IntermediateField F E
    hp : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem K x)) p
    ⊢ Iff (Eq (Algebra.adjoin F (p.rootSet E)) K.val.range) (Eq K.toSubalgebra (Al …
  -/
  rw [← K.range_val, eq_comm]
  /-
    🎉 no goals
  -/


theorem adjoin_rootSet_isSplittingField {p : F[X]} (hp : p.Splits (algebraMap F E)) :
    p.IsSplittingField F (adjoin F (p.rootSet E)) :=
  isSplittingField_iff.mpr ⟨splits_of_splits hp fun _ hx ↦ subset_adjoin F (p.rootSet E) hx, rfl⟩


theorem le_sup_toSubalgebra : E1.toSubalgebra ⊔ E2.toSubalgebra ≤ (E1 ⊔ E2).toSubalgebra :=
  sup_le (show E1 ≤ E1 ⊔ E2 from le_sup_left) (show E2 ≤ E1 ⊔ E2 from le_sup_right)


theorem sup_toSubalgebra_of_isAlgebraic_right [Algebra.IsAlgebraic K E2] :
    (E1 ⊔ E2).toSubalgebra = E1.toSubalgebra ⊔ E2.toSubalgebra := by
  have : (adjoin E1 (E2 : Set L)).toSubalgebra = _ := adjoin_algebraic_toSubalgebra fun x h ↦
    IsAlgebraic.tower_top E1 (isAlgebraic_iff.1
      (Algebra.IsAlgebraic.isAlgebraic (⟨x, h⟩ : E2)))
  /-
    K : Type u_3
    L : Type u_4
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    E1 E2 : IntermediateField K L
    inst✝ : Algebra.IsAlgebraic K (Subtype fun x => Membership.mem E2 x)
    this : Eq (IntermediateField.adjoin (Subtype fun x => Membership.mem E1 x) ↑E2 …
    ⊢ Eq (Max.max E1 E2).toSubalgebra (Max.max E1.toSubalgebra E2.toSubalgebra)
  -/
  apply_fun Subalgebra.restrictScalars K at this
  erw [← restrictScalars_toSubalgebra, restrictScalars_adjoin,
    Algebra.restrictScalars_adjoin] at this
  /-
    K : Type u_3
    L : Type u_4
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    E1 E2 : IntermediateField K L
    inst✝ : Algebra.IsAlgebraic K (Subtype fun x => Membership.mem E2 x)
    this : Eq (IntermediateField.adjoin K (Union.union ↑E1 ↑E2)).toSubalgebra (Alg …
    ⊢ Eq (Max.max E1 E2).toSubalgebra (Max.max E1.toSubalgebra E2.toSubalgebra)
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem sup_toSubalgebra_of_isAlgebraic_left [Algebra.IsAlgebraic K E1] :
    (E1 ⊔ E2).toSubalgebra = E1.toSubalgebra ⊔ E2.toSubalgebra := by
  /-
    K : Type u_3
    L : Type u_4
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    E1 E2 : IntermediateField K L
    inst✝ : Algebra.IsAlgebraic K (Subtype fun x => Membership.mem E1 x)
    ⊢ Eq (Max.max E1 E2).toSubalgebra (Max.max E1.toSubalgebra E2.toSubalgebra)
  -/
  have := sup_toSubalgebra_of_isAlgebraic_right E2 E1
  /-
    K : Type u_3
    L : Type u_4
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    E1 E2 : IntermediateField K L
    inst✝ : Algebra.IsAlgebraic K (Subtype fun x => Membership.mem E1 x)
    this : Eq (Max.max E2 E1).toSubalgebra (Max.max E2.toSubalgebra E1.toSubalgebra)
    ⊢ Eq (Max.max E1 E2).toSubalgebra (Max.max E1.toSubalgebra E2.toSubalgebra)
  -/
  rwa [sup_comm (a := E1), sup_comm (a := E1.toSubalgebra)]
  /-
    🎉 no goals
  -/


/-- The compositum of two intermediate fields is equal to the compositum of them
as subalgebras, if one of them is algebraic over the base field. -/
theorem sup_toSubalgebra_of_isAlgebraic
    (halg : Algebra.IsAlgebraic K E1 ∨ Algebra.IsAlgebraic K E2) :
    (E1 ⊔ E2).toSubalgebra = E1.toSubalgebra ⊔ E2.toSubalgebra :=
  halg.elim (fun _ ↦ sup_toSubalgebra_of_isAlgebraic_left E1 E2)
    (fun _ ↦ sup_toSubalgebra_of_isAlgebraic_right E1 E2)


theorem sup_toSubalgebra_of_left [FiniteDimensional K E1] :
    (E1 ⊔ E2).toSubalgebra = E1.toSubalgebra ⊔ E2.toSubalgebra :=
  sup_toSubalgebra_of_isAlgebraic_left E1 E2


theorem sup_toSubalgebra_of_right [FiniteDimensional K E2] :
    (E1 ⊔ E2).toSubalgebra = E1.toSubalgebra ⊔ E2.toSubalgebra :=
  sup_toSubalgebra_of_isAlgebraic_right E1 E2


instance finiteDimensional_sup [FiniteDimensional K E1] [FiniteDimensional K E2] :
    FiniteDimensional K (E1 ⊔ E2 : IntermediateField K L) := by
  /-
    F : Type u_1
    inst✝⁷ : Field F
    E : Type u_2
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    S : Set E
    α : E
    K : Type u_3
    L : Type u_4
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    E1 E2 : IntermediateField K L
    inst✝¹ : FiniteDimensional K (Subtype fun x => Membership.mem E1 x)
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem E2 x)
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (Max.max E1 E2) x)
  -/
  let g := Algebra.TensorProduct.productMap E1.val E2.val
  suffices g.range = (E1 ⊔ E2).toSubalgebra by
    have h : FiniteDimensional K (Subalgebra.toSubmodule g.range) :=
      g.toLinearMap.finiteDimensional_range
    rwa [this] at h
  /-
    F : Type u_1
    inst✝⁷ : Field F
    E : Type u_2
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    S : Set E
    α : E
    K : Type u_3
    L : Type u_4
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    E1 E2 : IntermediateField K L
    inst✝¹ : FiniteDimensional K (Subtype fun x => Membership.mem E1 x)
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem E2 x)
    g : AlgHom K (TensorProduct K (Subtype fun x => Membership.mem E1 x) (Subtype  …
    ⊢ Eq g.range (Max.max E1 E2).toSubalgebra
  -/
  rw [Algebra.TensorProduct.productMap_range, E1.range_val, E2.range_val, sup_toSubalgebra_of_left]
  /-
    🎉 no goals
  -/


/-- If `E1` and `E2` are intermediate fields, and at least one them are algebraic, then the rank of
the compositum of `E1` and `E2` is less than or equal to the product of that of `E1` and `E2`.
Note that this result is also true without algebraic assumption,
but the proof becomes very complicated. -/
theorem rank_sup_le_of_isAlgebraic
    (halg : Algebra.IsAlgebraic K E1 ∨ Algebra.IsAlgebraic K E2) :
    Module.rank K ↥(E1 ⊔ E2) ≤ Module.rank K E1 * Module.rank K E2 := by
  /-
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    E1 E2 : IntermediateField K L
    halg : Or (Algebra.IsAlgebraic K (Subtype fun x => Membership.mem E1 x)) (Alge …
    ⊢ LE.le (Module.rank K (Subtype fun x => Membership.mem (Max.max E1 E2) x)) (H …
  -/
  have := E1.toSubalgebra.rank_sup_le_of_free E2.toSubalgebra
  /-
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    E1 E2 : IntermediateField K L
    halg : Or (Algebra.IsAlgebraic K (Subtype fun x => Membership.mem E1 x)) (Alge …
    this : LE.le (Module.rank K (Subtype fun x => Membership.mem (Max.max E1.toSub …
    ⊢ LE.le (Module.rank K (Subtype fun x => Membership.mem (Max.max E1 E2) x)) (H …
  -/
  rwa [← sup_toSubalgebra_of_isAlgebraic E1 E2 halg] at this
  /-
    🎉 no goals
  -/


/-- If `E1` and `E2` are intermediate fields, then the `Module.finrank` of
the compositum of `E1` and `E2` is less than or equal to the product of that of `E1` and `E2`. -/
theorem finrank_sup_le :
    finrank K ↥(E1 ⊔ E2) ≤ finrank K E1 * finrank K E2 := by
  /-
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    E1 E2 : IntermediateField K L
    ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem (Max.max E1 E2) x)) …
  -/
  by_cases h : FiniteDimensional K E1
    /-
      case pos
      K : Type u_3
      L : Type u_4
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      E1 E2 : IntermediateField K L
      h : FiniteDimensional K (Subtype fun x => Membership.mem E1 x)
      ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem (Max.max E1 E2) x)) …
    -/
  · have := E1.toSubalgebra.finrank_sup_le_of_free E2.toSubalgebra
    /-
      case pos
      K : Type u_3
      L : Type u_4
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      E1 E2 : IntermediateField K L
      h : FiniteDimensional K (Subtype fun x => Membership.mem E1 x)
      this : LE.le (Module.finrank K (Subtype fun x => Membership.mem (Max.max E1.to …
      ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem (Max.max E1 E2) x)) …
    -/
    change _ ≤ finrank K E1 * finrank K E2 at this
    /-
      case pos
      K : Type u_3
      L : Type u_4
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      E1 E2 : IntermediateField K L
      h : FiniteDimensional K (Subtype fun x => Membership.mem E1 x)
      this : LE.le (Module.finrank K (Subtype fun x => Membership.mem (Max.max E1.to …
      ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem (Max.max E1 E2) x)) …
    -/
    rwa [← sup_toSubalgebra_of_left] at this
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    E1 E2 : IntermediateField K L
    h : Not (FiniteDimensional K (Subtype fun x => Membership.mem E1 x))
    ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem (Max.max E1 E2) x)) …
  -/
  rw [FiniteDimensional, ← rank_lt_aleph0_iff, not_lt] at h
  have := LinearMap.rank_le_of_injective _ <| Submodule.inclusion_injective <|
    show Subalgebra.toSubmodule E1.toSubalgebra ≤ Subalgebra.toSubmodule (E1 ⊔ E2).toSubalgebra by
      simp
  rw [show finrank K E1 = 0 from Cardinal.toNat_apply_of_aleph0_le h,
    show finrank K ↥(E1 ⊔ E2) = 0 from Cardinal.toNat_apply_of_aleph0_le (h.trans this), zero_mul]


theorem coe_iSup_of_directed [Nonempty ι] (dir : Directed (· ≤ ·) t) :
    ↑(iSup t) = ⋃ i, (t i : Set L) :=
  let M : IntermediateField K L :=
    { __ := Subalgebra.copy _ _ (Subalgebra.coe_iSup_of_directed dir).symm
      inv_mem' := fun _ hx ↦ have ⟨i, hi⟩ := Set.mem_iUnion.mp hx
        Set.mem_iUnion.mpr ⟨i, (t i).inv_mem hi⟩ }
  have : iSup t = M := le_antisymm
    (iSup_le fun i ↦ le_iSup (fun i ↦ (t i : Set L)) i) (Set.iUnion_subset fun _ ↦ le_iSup t _)
  this.symm ▸ rfl


theorem toSubalgebra_iSup_of_directed (dir : Directed (· ≤ ·) t) :
    (iSup t).toSubalgebra = ⨆ i, (t i).toSubalgebra := by
  /-
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ι : Type u_5
    t : ι → IntermediateField K L
    dir : Directed (fun x1 x2 => LE.le x1 x2) t
    ⊢ Eq (iSup t).toSubalgebra (iSup fun i => (t i).toSubalgebra)
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      K : Type u_3
      L : Type u_4
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      ι : Type u_5
      t : ι → IntermediateField K L
      dir : Directed (fun x1 x2 => LE.le x1 x2) t
      h✝ : IsEmpty ι
      ⊢ Eq (iSup t).toSubalgebra (iSup fun i => (t i).toSubalgebra)
    -/
  · simp_rw [iSup_of_empty, bot_toSubalgebra]
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type u_3
      L : Type u_4
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      ι : Type u_5
      t : ι → IntermediateField K L
      dir : Directed (fun x1 x2 => LE.le x1 x2) t
      h✝ : Nonempty ι
      ⊢ Eq (iSup t).toSubalgebra (iSup fun i => (t i).toSubalgebra)
    -/
  · exact SetLike.ext' ((coe_iSup_of_directed dir).trans (Subalgebra.coe_iSup_of_directed dir).symm)
    /-
      🎉 no goals
    -/


instance finiteDimensional_iSup_of_finite [h : Finite ι] [∀ i, FiniteDimensional K (t i)] :
    FiniteDimensional K (⨆ i, t i : IntermediateField K L) := by
  /-
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    S : Set E
    α : E
    K : Type u_3
    L : Type u_4
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    E1 E2 : IntermediateField K L
    ι : Type u_5
    t : ι → IntermediateField K L
    h : Finite ι
    inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (t i) x)
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (iSup fun i => t i) x)
  -/
  rw [← iSup_univ]
  /-
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    S : Set E
    α : E
    K : Type u_3
    L : Type u_4
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    E1 E2 : IntermediateField K L
    ι : Type u_5
    t : ι → IntermediateField K L
    h : Finite ι
    inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (t i) x)
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (iSup fun x => iSup fun …
  -/
  let P : Set ι → Prop := fun s => FiniteDimensional K (⨆ i ∈ s, t i : IntermediateField K L)
  /-
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    S : Set E
    α : E
    K : Type u_3
    L : Type u_4
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    E1 E2 : IntermediateField K L
    ι : Type u_5
    t : ι → IntermediateField K L
    h : Finite ι
    inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (t i) x)
    P : Set ι → Prop := fun s => FiniteDimensional K (Subtype fun x => Membership. …
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (iSup fun x => iSup fun …
  -/
  change P Set.univ
  /-
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    S : Set E
    α : E
    K : Type u_3
    L : Type u_4
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    E1 E2 : IntermediateField K L
    ι : Type u_5
    t : ι → IntermediateField K L
    h : Finite ι
    inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (t i) x)
    P : Set ι → Prop := fun s => FiniteDimensional K (Subtype fun x => Membership. …
    ⊢ P Set.univ
  -/
  apply Set.Finite.induction_on
  /-
    case h
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    S : Set E
    α : E
    K : Type u_3
    L : Type u_4
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    E1 E2 : IntermediateField K L
    ι : Type u_5
    t : ι → IntermediateField K L
    h : Finite ι
    inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (t i) x)
    P : Set ι → Prop := fun s => FiniteDimensional K (Subtype fun x => Membership. …
    ⊢ Set.univ.Finite
  -/
  all_goals dsimp only [P]
    /-
      case h
      F : Type u_1
      inst✝⁶ : Field F
      E : Type u_2
      inst✝⁵ : Field E
      inst✝⁴ : Algebra F E
      S : Set E
      α : E
      K : Type u_3
      L : Type u_4
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      E1 E2 : IntermediateField K L
      ι : Type u_5
      t : ι → IntermediateField K L
      h : Finite ι
      inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (t i) x)
      P : Set ι → Prop := fun s => FiniteDimensional K (Subtype fun x => Membership. …
      ⊢ Set.univ.Finite
    -/
  · exact Set.finite_univ
    /-
      🎉 no goals
    -/
    /-
      case H0
      F : Type u_1
      inst✝⁶ : Field F
      E : Type u_2
      inst✝⁵ : Field E
      inst✝⁴ : Algebra F E
      S : Set E
      α : E
      K : Type u_3
      L : Type u_4
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      E1 E2 : IntermediateField K L
      ι : Type u_5
      t : ι → IntermediateField K L
      h : Finite ι
      inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (t i) x)
      P : Set ι → Prop := fun s => FiniteDimensional K (Subtype fun x => Membership. …
      ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (iSup fun i => iSup fun …
    -/
  · rw [iSup_emptyset]
    /-
      case H0
      F : Type u_1
      inst✝⁶ : Field F
      E : Type u_2
      inst✝⁵ : Field E
      inst✝⁴ : Algebra F E
      S : Set E
      α : E
      K : Type u_3
      L : Type u_4
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      E1 E2 : IntermediateField K L
      ι : Type u_5
      t : ι → IntermediateField K L
      h : Finite ι
      inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (t i) x)
      P : Set ι → Prop := fun s => FiniteDimensional K (Subtype fun x => Membership. …
      ⊢ FiniteDimensional K (Subtype fun x => Membership.mem Bot.bot x)
    -/
    exact (botEquiv K L).symm.toLinearEquiv.finiteDimensional
    /-
      🎉 no goals
    -/
    /-
      case H1
      F : Type u_1
      inst✝⁶ : Field F
      E : Type u_2
      inst✝⁵ : Field E
      inst✝⁴ : Algebra F E
      S : Set E
      α : E
      K : Type u_3
      L : Type u_4
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      E1 E2 : IntermediateField K L
      ι : Type u_5
      t : ι → IntermediateField K L
      h : Finite ι
      inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (t i) x)
      P : Set ι → Prop := fun s => FiniteDimensional K (Subtype fun x => Membership. …
      ⊢ ∀ {a : ι} {s : Set ι}, Not (Membership.mem s a) → s.Finite → FiniteDimension …
    -/
  · intro _ s _ _ hs
    /-
      case H1
      F : Type u_1
      inst✝⁶ : Field F
      E : Type u_2
      inst✝⁵ : Field E
      inst✝⁴ : Algebra F E
      S : Set E
      α : E
      K : Type u_3
      L : Type u_4
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      E1 E2 : IntermediateField K L
      ι : Type u_5
      t : ι → IntermediateField K L
      h : Finite ι
      inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (t i) x)
      P : Set ι → Prop := fun s => FiniteDimensional K (Subtype fun x => Membership. …
      a✝² : ι
      s : Set ι
      a✝¹ : Not (Membership.mem s a✝²)
      a✝ : s.Finite
      hs : FiniteDimensional K (Subtype fun x => Membership.mem (iSup fun i => iSup  …
      ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (iSup fun i => iSup fun …
    -/
    rw [iSup_insert]
    /-
      case H1
      F : Type u_1
      inst✝⁶ : Field F
      E : Type u_2
      inst✝⁵ : Field E
      inst✝⁴ : Algebra F E
      S : Set E
      α : E
      K : Type u_3
      L : Type u_4
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      E1 E2 : IntermediateField K L
      ι : Type u_5
      t : ι → IntermediateField K L
      h : Finite ι
      inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (t i) x)
      P : Set ι → Prop := fun s => FiniteDimensional K (Subtype fun x => Membership. …
      a✝² : ι
      s : Set ι
      a✝¹ : Not (Membership.mem s a✝²)
      a✝ : s.Finite
      hs : FiniteDimensional K (Subtype fun x => Membership.mem (iSup fun i => iSup  …
      ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (Max.max (t a✝²) (iSup  …
    -/
    exact IntermediateField.finiteDimensional_sup _ _
    /-
      🎉 no goals
    -/


instance finiteDimensional_iSup_of_finset
    /- Porting note: changed `h` from `∀ i ∈ s, FiniteDimensional K (t i)` because this caused an
      error. See `finiteDimensional_iSup_of_finset'` for a stronger version, that was the one
      used in mathlib3. -/
    {s : Finset ι} [∀ i, FiniteDimensional K (t i)] :
    FiniteDimensional K (⨆ i ∈ s, t i : IntermediateField K L) :=
  iSup_subtype'' s t ▸ IntermediateField.finiteDimensional_iSup_of_finite


theorem finiteDimensional_iSup_of_finset'
    /- Porting note: this was the mathlib3 version. Using `[h : ...]`, as in mathlib3, causes the
    error "invalid parametric local instance". -/
    {s : Finset ι} (h : ∀ i ∈ s, FiniteDimensional K (t i)) :
    FiniteDimensional K (⨆ i ∈ s, t i : IntermediateField K L) :=
  have := Subtype.forall'.mp h
  iSup_subtype'' s t ▸ IntermediateField.finiteDimensional_iSup_of_finite


/-- A compositum of splitting fields is a splitting field -/
theorem isSplittingField_iSup {p : ι → K[X]}
    {s : Finset ι} (h0 : ∏ i ∈ s, p i ≠ 0) (h : ∀ i ∈ s, (p i).IsSplittingField K (t i)) :
    (∏ i ∈ s, p i).IsSplittingField K (⨆ i ∈ s, t i : IntermediateField K L) := by
  /-
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ι : Type u_5
    t : ι → IntermediateField K L
    p : ι → Polynomial K
    s : Finset ι
    h0 : Ne (s.prod fun i => p i) 0
    h : ∀ (i : ι), Membership.mem s i → Polynomial.IsSplittingField K (Subtype fun …
    ⊢ Polynomial.IsSplittingField K (Subtype fun x => Membership.mem (iSup fun i = …
  -/
  let F : IntermediateField K L := ⨆ i ∈ s, t i
  /-
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ι : Type u_5
    t : ι → IntermediateField K L
    p : ι → Polynomial K
    s : Finset ι
    h0 : Ne (s.prod fun i => p i) 0
    h : ∀ (i : ι), Membership.mem s i → Polynomial.IsSplittingField K (Subtype fun …
    F : IntermediateField K L := iSup fun i => iSup fun h => t i
    ⊢ Polynomial.IsSplittingField K (Subtype fun x => Membership.mem (iSup fun i = …
  -/
  have hF : ∀ i ∈ s, t i ≤ F := fun i hi ↦ le_iSup_of_le i (le_iSup (fun _ ↦ t i) hi)
  /-
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ι : Type u_5
    t : ι → IntermediateField K L
    p : ι → Polynomial K
    s : Finset ι
    h0 : Ne (s.prod fun i => p i) 0
    h : ∀ (i : ι), Membership.mem s i → Polynomial.IsSplittingField K (Subtype fun …
    F : IntermediateField K L := iSup fun i => iSup fun h => t i
    hF : ∀ (i : ι), Membership.mem s i → LE.le (t i) F
    ⊢ Polynomial.IsSplittingField K (Subtype fun x => Membership.mem (iSup fun i = …
  -/
  simp only [isSplittingField_iff] at h ⊢
  refine
    ⟨splits_prod (algebraMap K F) fun i hi ↦
        splits_comp_of_splits (algebraMap K (t i)) (inclusion (hF i hi)).toRingHom
          (h i hi).1,
      ?_⟩
  /-
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ι : Type u_5
    t : ι → IntermediateField K L
    p : ι → Polynomial K
    s : Finset ι
    h0 : Ne (s.prod fun i => p i) 0
    F : IntermediateField K L := iSup fun i => iSup fun h => t i
    hF : ∀ (i : ι), Membership.mem s i → LE.le (t i) F
    h : ∀ (i : ι), Membership.mem s i → And (Polynomial.Splits (algebraMap K (Subt …
    ⊢ Eq (iSup fun i => iSup fun h => t i) (IntermediateField.adjoin K ((s.prod fu …
  -/
  simp only [rootSet_prod p s h0, ← Set.iSup_eq_iUnion, (@gc K _ L _ _).l_iSup₂]
  /-
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ι : Type u_5
    t : ι → IntermediateField K L
    p : ι → Polynomial K
    s : Finset ι
    h0 : Ne (s.prod fun i => p i) 0
    F : IntermediateField K L := iSup fun i => iSup fun h => t i
    hF : ∀ (i : ι), Membership.mem s i → LE.le (t i) F
    h : ∀ (i : ι), Membership.mem s i → And (Polynomial.Splits (algebraMap K (Subt …
    ⊢ Eq (iSup fun i => iSup fun h => t i) (iSup fun i => iSup fun j => Intermedia …
  -/
  exact iSup_congr fun i ↦ iSup_congr fun hi ↦ (h i hi).2
  /-
    🎉 no goals
  -/


/-- If `K / E / F` is a field extension tower, `L` is an intermediate field of `K / F`, such that
either `E / F` or `L / F` is algebraic, then `E(L) = E[L]`. -/
theorem adjoin_toSubalgebra_of_isAlgebraic (L : IntermediateField F K)
    (halg : Algebra.IsAlgebraic F E ∨ Algebra.IsAlgebraic F L) :
    (adjoin E (L : Set K)).toSubalgebra = Algebra.adjoin E (L : Set K) := by
  /-
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    L : IntermediateField F K
    halg : Or (Algebra.IsAlgebraic F E) (Algebra.IsAlgebraic F (Subtype fun x => M …
    ⊢ Eq (IntermediateField.adjoin E ↑L).toSubalgebra (Algebra.adjoin E ↑L)
  -/
  let i := IsScalarTower.toAlgHom F E K
  /-
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    L : IntermediateField F K
    halg : Or (Algebra.IsAlgebraic F E) (Algebra.IsAlgebraic F (Subtype fun x => M …
    i : AlgHom F E K := IsScalarTower.toAlgHom F E K
    ⊢ Eq (IntermediateField.adjoin E ↑L).toSubalgebra (Algebra.adjoin E ↑L)
  -/
  let E' := i.fieldRange
  /-
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    L : IntermediateField F K
    halg : Or (Algebra.IsAlgebraic F E) (Algebra.IsAlgebraic F (Subtype fun x => M …
    i : AlgHom F E K := IsScalarTower.toAlgHom F E K
    E' : IntermediateField F K := i.fieldRange
    ⊢ Eq (IntermediateField.adjoin E ↑L).toSubalgebra (Algebra.adjoin E ↑L)
  -/
  let i' : E ≃ₐ[F] E' := AlgEquiv.ofInjectiveField i
  /-
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    L : IntermediateField F K
    halg : Or (Algebra.IsAlgebraic F E) (Algebra.IsAlgebraic F (Subtype fun x => M …
    i : AlgHom F E K := IsScalarTower.toAlgHom F E K
    E' : IntermediateField F K := i.fieldRange
    i' : AlgEquiv F E (Subtype fun x => Membership.mem E' x) := AlgEquiv.ofInjecti …
    ⊢ Eq (IntermediateField.adjoin E ↑L).toSubalgebra (Algebra.adjoin E ↑L)
  -/
  have hi : algebraMap E K = (algebraMap E' K) ∘ i' := by ext x; rfl
  /-
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    L : IntermediateField F K
    halg : Or (Algebra.IsAlgebraic F E) (Algebra.IsAlgebraic F (Subtype fun x => M …
    i : AlgHom F E K := IsScalarTower.toAlgHom F E K
    E' : IntermediateField F K := i.fieldRange
    i' : AlgEquiv F E (Subtype fun x => Membership.mem E' x) := AlgEquiv.ofInjecti …
    hi : Eq (⇑(algebraMap E K)) (Function.comp ⇑(algebraMap (Subtype fun x => Memb …
    ⊢ Eq (IntermediateField.adjoin E ↑L).toSubalgebra (Algebra.adjoin E ↑L)
  -/
  apply_fun _ using Subalgebra.restrictScalars_injective F
  erw [← restrictScalars_toSubalgebra, restrictScalars_adjoin_of_algEquiv i' hi,
    Algebra.restrictScalars_adjoin_of_algEquiv i' hi, restrictScalars_adjoin,
    Algebra.restrictScalars_adjoin]
  exact E'.sup_toSubalgebra_of_isAlgebraic L (halg.imp
    (fun (_ : Algebra.IsAlgebraic F E) ↦ i'.isAlgebraic) id)


theorem adjoin_toSubalgebra_of_isAlgebraic_left (L : IntermediateField F K)
    [halg : Algebra.IsAlgebraic F E] :
    (adjoin E (L : Set K)).toSubalgebra = Algebra.adjoin E (L : Set K) :=
  adjoin_toSubalgebra_of_isAlgebraic E L (Or.inl halg)


theorem adjoin_toSubalgebra_of_isAlgebraic_right (L : IntermediateField F K)
    [halg : Algebra.IsAlgebraic F L] :
    (adjoin E (L : Set K)).toSubalgebra = Algebra.adjoin E (L : Set K) :=
  adjoin_toSubalgebra_of_isAlgebraic E L (Or.inr halg)


/-- If `K / E / F` is a field extension tower, `L` is an intermediate field of `K / F`, such that
either `E / F` or `L / F` is algebraic, then `[E(L) : E] ≤ [L : F]`. A corollary of
`Subalgebra.adjoin_rank_le` since in this case `E(L) = E[L]`. -/
theorem adjoin_rank_le_of_isAlgebraic (L : IntermediateField F K)
    (halg : Algebra.IsAlgebraic F E ∨ Algebra.IsAlgebraic F L) :
    Module.rank E (adjoin E (L : Set K)) ≤ Module.rank F L := by
  have h : (adjoin E (L.toSubalgebra : Set K)).toSubalgebra =
      Algebra.adjoin E (L.toSubalgebra : Set K) :=
    L.adjoin_toSubalgebra_of_isAlgebraic E halg
  /-
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    L : IntermediateField F K
    halg : Or (Algebra.IsAlgebraic F E) (Algebra.IsAlgebraic F (Subtype fun x => M …
    h : Eq (IntermediateField.adjoin E ↑L.toSubalgebra).toSubalgebra (Algebra.adjo …
    ⊢ LE.le (Module.rank E (Subtype fun x => Membership.mem (IntermediateField.adj …
  -/
  have := L.toSubalgebra.adjoin_rank_le E
  /-
    F : Type u_1
    inst✝⁶ : Field F
    E : Type u_2
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    L : IntermediateField F K
    halg : Or (Algebra.IsAlgebraic F E) (Algebra.IsAlgebraic F (Subtype fun x => M …
    h : Eq (IntermediateField.adjoin E ↑L.toSubalgebra).toSubalgebra (Algebra.adjo …
    this : LE.le (Module.rank E (Subtype fun x => Membership.mem (Algebra.adjoin E …
    ⊢ LE.le (Module.rank E (Subtype fun x => Membership.mem (IntermediateField.adj …
  -/
  rwa [(Subalgebra.equivOfEq _ _ h).symm.toLinearEquiv.rank_eq] at this
  /-
    🎉 no goals
  -/


theorem adjoin_rank_le_of_isAlgebraic_left (L : IntermediateField F K)
    [halg : Algebra.IsAlgebraic F E] :
    Module.rank E (adjoin E (L : Set K)) ≤ Module.rank F L :=
  adjoin_rank_le_of_isAlgebraic E L (Or.inl halg)


theorem adjoin_rank_le_of_isAlgebraic_right (L : IntermediateField F K)
    [halg : Algebra.IsAlgebraic F L] :
    Module.rank E (adjoin E (L : Set K)) ≤ Module.rank F L :=
  adjoin_rank_le_of_isAlgebraic E L (Or.inr halg)


                                                                                  /-
                                                                                    F : Type u_1
                                                                                    inst✝² : Field F
                                                                                    E : Type u_2
                                                                                    inst✝¹ : Field E
                                                                                    inst✝ : Algebra F E
                                                                                    α : E
                                                                                    K : IntermediateField F E
                                                                                    ⊢ Iff (LE.le (IntermediateField.adjoin F (Singleton.singleton α)) K) (Membersh …
                                                                                  -/
theorem adjoin_simple_le_iff {K : IntermediateField F E} : F⟮α⟯ ≤ K ↔ α ∈ K := by simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem biSup_adjoin_simple : ⨆ x ∈ S, F⟮x⟯ = adjoin F S := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    ⊢ Eq (iSup fun x => iSup fun h => IntermediateField.adjoin F (Singleton.single …
  -/
  rw [← iSup_subtype'', ← gc.l_iSup, iSup_subtype'']; congr; exact S.biUnion_of_singleton
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Adjoining a single element is compact in the lattice of intermediate fields. -/
theorem adjoin_simple_isCompactElement (x : E) : IsCompactElement F⟮x⟯ := by
  simp_rw [isCompactElement_iff_le_of_directed_sSup_le,
    adjoin_simple_le_iff, sSup_eq_iSup', ← exists_prop]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x : E
    ⊢ ∀ (s : Set (IntermediateField F E)), s.Nonempty → DirectedOn (fun x1 x2 => L …
  -/
  intro s hne hs hx
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x : E
    s : Set (IntermediateField F E)
    hne : s.Nonempty
    hs : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hx : Membership.mem (iSup fun a => ↑a) x
    ⊢ Exists fun x_1 => Exists fun _h => Membership.mem x_1 x
  -/
  have := hne.to_subtype
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x : E
    s : Set (IntermediateField F E)
    hne : s.Nonempty
    hs : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hx : Membership.mem (iSup fun a => ↑a) x
    this : Nonempty ↑s
    ⊢ Exists fun x_1 => Exists fun _h => Membership.mem x_1 x
  -/
  rwa [← SetLike.mem_coe, coe_iSup_of_directed hs.directed_val, mem_iUnion, Subtype.exists] at hx
  /-
    🎉 no goals
  -/

-- Porting note: original proof times out.

/-- Adjoining a finite subset is compact in the lattice of intermediate fields. -/
theorem adjoin_finset_isCompactElement (S : Finset E) :
    IsCompactElement (adjoin F S : IntermediateField F E) := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Finset E
    ⊢ CompleteLattice.IsCompactElement (IntermediateField.adjoin F ↑S)
  -/
  rw [← biSup_adjoin_simple]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Finset E
    ⊢ CompleteLattice.IsCompactElement (iSup fun x => iSup fun h => IntermediateFi …
  -/
  simp_rw [Finset.mem_coe, ← Finset.sup_eq_iSup]
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Finset E
    ⊢ CompleteLattice.IsCompactElement (S.sup fun a => IntermediateField.adjoin F  …
  -/
  exact isCompactElement_finsetSup S fun x _ => adjoin_simple_isCompactElement x
  /-
    🎉 no goals
  -/


/-- Adjoining a finite subset is compact in the lattice of intermediate fields. -/
theorem adjoin_finite_isCompactElement {S : Set E} (h : S.Finite) : IsCompactElement (adjoin F S) :=
  Finite.coe_toFinset h ▸ adjoin_finset_isCompactElement h.toFinset


/-- The lattice of intermediate fields is compactly generated. -/
instance : IsCompactlyGenerated (IntermediateField F E) :=
  ⟨fun s =>
    ⟨(fun x => F⟮x⟯) '' s,
          /-
            F : Type u_1
            inst✝² : Field F
            E : Type u_2
            inst✝¹ : Field E
            inst✝ : Algebra F E
            S : Set E
            α : E
            s : IntermediateField F E
            ⊢ ∀ (x : IntermediateField F E), Membership.mem (Set.image (fun x => Intermedi …
          -/
      ⟨by rintro t ⟨x, _, rfl⟩; exact adjoin_simple_isCompactElement x,
                                /-
                                  🎉 no goals
                                -/
        sSup_image.trans <| (biSup_adjoin_simple _).trans <|
          le_antisymm (adjoin_le_iff.mpr le_rfl) <| subset_adjoin F (s : Set E)⟩⟩⟩


theorem exists_finset_of_mem_iSup {ι : Type*} {f : ι → IntermediateField F E} {x : E}
    (hx : x ∈ ⨆ i, f i) : ∃ s : Finset ι, x ∈ ⨆ i ∈ s, f i := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ι : Type u_3
    f : ι → IntermediateField F E
    x : E
    hx : Membership.mem (iSup fun i => f i) x
    ⊢ Exists fun s => Membership.mem (iSup fun i => iSup fun h => f i) x
  -/
  have := (adjoin_simple_isCompactElement x).exists_finset_of_le_iSup (IntermediateField F E) f
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ι : Type u_3
    f : ι → IntermediateField F E
    x : E
    hx : Membership.mem (iSup fun i => f i) x
    this : LE.le (IntermediateField.adjoin F (Singleton.singleton x)) (iSup fun i  …
    ⊢ Exists fun s => Membership.mem (iSup fun i => iSup fun h => f i) x
  -/
  simp only [adjoin_simple_le_iff] at this
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ι : Type u_3
    f : ι → IntermediateField F E
    x : E
    hx : Membership.mem (iSup fun i => f i) x
    this : Membership.mem (iSup fun i => f i) x → Exists fun s => Membership.mem ( …
    ⊢ Exists fun s => Membership.mem (iSup fun i => iSup fun h => f i) x
  -/
  exact this hx
  /-
    🎉 no goals
  -/


theorem exists_finset_of_mem_supr' {ι : Type*} {f : ι → IntermediateField F E} {x : E}
    (hx : x ∈ ⨆ i, f i) : ∃ s : Finset (Σ i, f i), x ∈ ⨆ i ∈ s, F⟮(i.2 : E)⟯ := by
-- Porting note: writing `fun i x h => ...` does not work.
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ι : Type u_3
    f : ι → IntermediateField F E
    x : E
    hx : Membership.mem (iSup fun i => f i) x
    ⊢ Exists fun s => Membership.mem (iSup fun i => iSup fun h => IntermediateFiel …
  -/
  refine exists_finset_of_mem_iSup (SetLike.le_def.mp (iSup_le fun i ↦ ?_) hx)
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ι : Type u_3
    f : ι → IntermediateField F E
    x : E
    hx : Membership.mem (iSup fun i => f i) x
    i : ι
    ⊢ LE.le (f i) (iSup fun i => IntermediateField.adjoin F (Singleton.singleton ↑ …
  -/
  exact fun x h ↦ SetLike.le_def.mp (le_iSup_of_le ⟨i, x, h⟩ (by simp)) (mem_adjoin_simple_self F x)
  /-
    🎉 no goals
  -/


theorem exists_finset_of_mem_supr'' {ι : Type*} {f : ι → IntermediateField F E}
    (h : ∀ i, Algebra.IsAlgebraic F (f i)) {x : E} (hx : x ∈ ⨆ i, f i) :
    ∃ s : Finset (Σ i, f i), x ∈ ⨆ i ∈ s, adjoin F ((minpoly F (i.2 : _)).rootSet E) := by
-- Porting note: writing `fun i x1 hx1 => ...` does not work.
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ι : Type u_3
    f : ι → IntermediateField F E
    h : ∀ (i : ι), Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (f i) x)
    x : E
    hx : Membership.mem (iSup fun i => f i) x
    ⊢ Exists fun s => Membership.mem (iSup fun i => iSup fun h => IntermediateFiel …
  -/
  refine exists_finset_of_mem_iSup (SetLike.le_def.mp (iSup_le (fun i => ?_)) hx)
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ι : Type u_3
    f : ι → IntermediateField F E
    h : ∀ (i : ι), Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (f i) x)
    x : E
    hx : Membership.mem (iSup fun i => f i) x
    i : ι
    ⊢ LE.le (f i) (iSup fun i => IntermediateField.adjoin F ((minpoly F i.snd).roo …
  -/
  intro x1 hx1
  refine SetLike.le_def.mp (le_iSup_of_le ⟨i, x1, hx1⟩ ?_)
    (subset_adjoin F (rootSet (minpoly F x1) E) ?_)
    /-
      case refine_1
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ι : Type u_3
      f : ι → IntermediateField F E
      h : ∀ (i : ι), Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (f i) x)
      x : E
      hx : Membership.mem (iSup fun i => f i) x
      i : ι
      x1 : E
      hx1 : Membership.mem (f i) x1
      ⊢ LE.le (IntermediateField.adjoin F ((minpoly F x1).rootSet E)) (IntermediateF …
    -/
  · rw [IntermediateField.minpoly_eq, Subtype.coe_mk]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ι : Type u_3
      f : ι → IntermediateField F E
      h : ∀ (i : ι), Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (f i) x)
      x : E
      hx : Membership.mem (iSup fun i => f i) x
      i : ι
      x1 : E
      hx1 : Membership.mem (f i) x1
      ⊢ Membership.mem ((minpoly F x1).rootSet E) x1
    -/
  · rw [mem_rootSet_of_ne, minpoly.aeval]
    /-
      case refine_2.hp
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ι : Type u_3
      f : ι → IntermediateField F E
      h : ∀ (i : ι), Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (f i) x)
      x : E
      hx : Membership.mem (iSup fun i => f i) x
      i : ι
      x1 : E
      hx1 : Membership.mem (f i) x1
      ⊢ Ne (minpoly F x1) 0
    -/
    exact minpoly.ne_zero (isIntegral_iff.mp (Algebra.IsIntegral.isIntegral (⟨x1, hx1⟩ : f i)))
    /-
      🎉 no goals
    -/


theorem exists_finset_of_mem_adjoin {S : Set E} {x : E} (hx : x ∈ adjoin F S) :
    ∃ T : Finset E, (T : Set E) ⊆ S ∧ x ∈ adjoin F (T : Set E) := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    x : E
    hx : Membership.mem (IntermediateField.adjoin F S) x
    ⊢ Exists fun T => And (HasSubset.Subset (↑T) S) (Membership.mem (IntermediateF …
  -/
  simp_rw [← biSup_adjoin_simple S, ← iSup_subtype''] at hx
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    x : E
    hx : Membership.mem (iSup fun i => IntermediateField.adjoin F (Singleton.singl …
    ⊢ Exists fun T => And (HasSubset.Subset (↑T) S) (Membership.mem (IntermediateF …
  -/
  obtain ⟨s, hx'⟩ := exists_finset_of_mem_iSup hx
  classical
  refine ⟨s.image Subtype.val, by simp, SetLike.le_def.mp ?_ hx'⟩
  simp_rw [Finset.coe_image, iSup_le_iff, adjoin_le_iff]
  rintro _ h _ rfl
  exact subset_adjoin F _ ⟨_, h, rfl⟩


@[simp]
theorem adjoin_eq_bot_iff : adjoin F S = ⊥ ↔ S ⊆ (⊥ : IntermediateField F E) := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : Set E
    ⊢ Iff (Eq (IntermediateField.adjoin F S) Bot.bot) (HasSubset.Subset S ↑Bot.bot)
  -/
  rw [eq_bot_iff, adjoin_le_iff]
  /-
    🎉 no goals
  -/

/- Porting note: this was tagged `simp`. -/

theorem adjoin_simple_eq_bot_iff : F⟮α⟯ = ⊥ ↔ α ∈ (⊥ : IntermediateField F E) := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    ⊢ Iff (Eq (IntermediateField.adjoin F (Singleton.singleton α)) Bot.bot) (Membe …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem adjoin_zero : F⟮(0 : E)⟯ = ⊥ :=
  adjoin_simple_eq_bot_iff.mpr (zero_mem ⊥)


@[simp]
theorem adjoin_one : F⟮(1 : E)⟯ = ⊥ :=
  adjoin_simple_eq_bot_iff.mpr (one_mem ⊥)


@[simp]
theorem adjoin_intCast (n : ℤ) : F⟮(n : E)⟯ = ⊥ := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    n : Int
    ⊢ Eq (IntermediateField.adjoin F (Singleton.singleton ↑n)) Bot.bot
  -/
  exact adjoin_simple_eq_bot_iff.mpr (intCast_mem ⊥ n)
  /-
    🎉 no goals
  -/


@[simp]
theorem adjoin_natCast (n : ℕ) : F⟮(n : E)⟯ = ⊥ :=
  adjoin_simple_eq_bot_iff.mpr (natCast_mem ⊥ n)


@[deprecated (since := "2024-04-05")] alias adjoin_int := adjoin_intCast

@[deprecated (since := "2024-04-05")] alias adjoin_nat := adjoin_natCast


@[simp]
theorem rank_eq_one_iff : Module.rank F K = 1 ↔ K = ⊥ := by
  rw [← toSubalgebra_inj, ← rank_eq_rank_subalgebra, Subalgebra.rank_eq_one_iff,
    bot_toSubalgebra]


@[simp]
theorem finrank_eq_one_iff : finrank F K = 1 ↔ K = ⊥ := by
  rw [← toSubalgebra_inj, ← finrank_eq_finrank_subalgebra, Subalgebra.finrank_eq_one_iff,
    bot_toSubalgebra]


@[simp]
protected theorem rank_bot : Module.rank F (⊥ : IntermediateField F E) = 1 := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Eq (Module.rank F (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  rw [rank_eq_one_iff]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem finrank_bot : finrank F (⊥ : IntermediateField F E) = 1 := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Eq (Module.finrank F (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  rw [finrank_eq_one_iff]
  /-
    🎉 no goals
  -/


@[simp] theorem rank_bot' : Module.rank (⊥ : IntermediateField F E) E = Module.rank F E := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Eq (Module.rank (Subtype fun x => Membership.mem Bot.bot x) E) (Module.rank  …
  -/
  rw [← rank_mul_rank F (⊥ : IntermediateField F E) E, IntermediateField.rank_bot, one_mul]
  /-
    🎉 no goals
  -/


@[simp] theorem finrank_bot' : finrank (⊥ : IntermediateField F E) E = finrank F E :=
  congr(Cardinal.toNat $(rank_bot'))


@[simp] protected theorem rank_top : Module.rank (⊤ : IntermediateField F E) E = 1 :=
  Subalgebra.bot_eq_top_iff_rank_eq_one.mp <| top_le_iff.mp fun x _ ↦ ⟨⟨x, trivial⟩, rfl⟩


@[simp] protected theorem finrank_top : finrank (⊤ : IntermediateField F E) E = 1 :=
  rank_eq_one_iff_finrank_eq_one.mp IntermediateField.rank_top


@[simp] theorem rank_top' : Module.rank F (⊤ : IntermediateField F E) = Module.rank F E :=
  rank_top F E


@[simp] theorem finrank_top' : finrank F (⊤ : IntermediateField F E) = finrank F E :=
  finrank_top F E


theorem rank_adjoin_eq_one_iff : Module.rank F (adjoin F S) = 1 ↔ S ⊆ (⊥ : IntermediateField F E) :=
  Iff.trans rank_eq_one_iff adjoin_eq_bot_iff


theorem rank_adjoin_simple_eq_one_iff :
    Module.rank F F⟮α⟯ = 1 ↔ α ∈ (⊥ : IntermediateField F E) := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    ⊢ Iff (Eq (Module.rank F (Subtype fun x => Membership.mem (IntermediateField.a …
  -/
  rw [rank_adjoin_eq_one_iff]; exact Set.singleton_subset_iff
                               /-
                                 🎉 no goals
                               -/


theorem finrank_adjoin_eq_one_iff : finrank F (adjoin F S) = 1 ↔ S ⊆ (⊥ : IntermediateField F E) :=
  Iff.trans finrank_eq_one_iff adjoin_eq_bot_iff


theorem finrank_adjoin_simple_eq_one_iff :
    finrank F F⟮α⟯ = 1 ↔ α ∈ (⊥ : IntermediateField F E) := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    ⊢ Iff (Eq (Module.finrank F (Subtype fun x => Membership.mem (IntermediateFiel …
  -/
  rw [finrank_adjoin_eq_one_iff]; exact Set.singleton_subset_iff
                                  /-
                                    🎉 no goals
                                  -/


/-- If `F⟮x⟯` has dimension `1` over `F` for every `x ∈ E` then `F = E`. -/
theorem bot_eq_top_of_rank_adjoin_eq_one (h : ∀ x : E, Module.rank F F⟮x⟯ = 1) :
    (⊥ : IntermediateField F E) = ⊤ := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : ∀ (x : E), Eq (Module.rank F (Subtype fun x_1 => Membership.mem (Intermedi …
    ⊢ Eq Bot.bot Top.top
  -/
  ext y
  /-
    case h
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : ∀ (x : E), Eq (Module.rank F (Subtype fun x_1 => Membership.mem (Intermedi …
    y : E
    ⊢ Iff (Membership.mem Bot.bot y) (Membership.mem Top.top y)
  -/
  rw [iff_true_right IntermediateField.mem_top]
  /-
    case h
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : ∀ (x : E), Eq (Module.rank F (Subtype fun x_1 => Membership.mem (Intermedi …
    y : E
    ⊢ Membership.mem Bot.bot y
  -/
  exact rank_adjoin_simple_eq_one_iff.mp (h y)
  /-
    🎉 no goals
  -/


theorem bot_eq_top_of_finrank_adjoin_eq_one (h : ∀ x : E, finrank F F⟮x⟯ = 1) :
    (⊥ : IntermediateField F E) = ⊤ := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : ∀ (x : E), Eq (Module.finrank F (Subtype fun x_1 => Membership.mem (Interm …
    ⊢ Eq Bot.bot Top.top
  -/
  ext y
  /-
    case h
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : ∀ (x : E), Eq (Module.finrank F (Subtype fun x_1 => Membership.mem (Interm …
    y : E
    ⊢ Iff (Membership.mem Bot.bot y) (Membership.mem Top.top y)
  -/
  rw [iff_true_right IntermediateField.mem_top]
  /-
    case h
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : ∀ (x : E), Eq (Module.finrank F (Subtype fun x_1 => Membership.mem (Interm …
    y : E
    ⊢ Membership.mem Bot.bot y
  -/
  exact finrank_adjoin_simple_eq_one_iff.mp (h y)
  /-
    🎉 no goals
  -/


theorem subsingleton_of_rank_adjoin_eq_one (h : ∀ x : E, Module.rank F F⟮x⟯ = 1) :
    Subsingleton (IntermediateField F E) :=
  subsingleton_of_bot_eq_top (bot_eq_top_of_rank_adjoin_eq_one h)


theorem subsingleton_of_finrank_adjoin_eq_one (h : ∀ x : E, finrank F F⟮x⟯ = 1) :
    Subsingleton (IntermediateField F E) :=
  subsingleton_of_bot_eq_top (bot_eq_top_of_finrank_adjoin_eq_one h)


/-- If `F⟮x⟯` has dimension `≤1` over `F` for every `x ∈ E` then `F = E`. -/
theorem bot_eq_top_of_finrank_adjoin_le_one [FiniteDimensional F E]
    (h : ∀ x : E, finrank F F⟮x⟯ ≤ 1) : (⊥ : IntermediateField F E) = ⊤ := by
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    h : ∀ (x : E), LE.le (Module.finrank F (Subtype fun x_1 => Membership.mem (Int …
    ⊢ Eq Bot.bot Top.top
  -/
  apply bot_eq_top_of_finrank_adjoin_eq_one
  /-
    case h
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    h : ∀ (x : E), LE.le (Module.finrank F (Subtype fun x_1 => Membership.mem (Int …
    ⊢ ∀ (x : E), Eq (Module.finrank F (Subtype fun x_1 => Membership.mem (Intermed …
  -/
  exact fun x => by linarith [h x, show 0 < finrank F F⟮x⟯ from finrank_pos]
  /-
    🎉 no goals
  -/


theorem subsingleton_of_finrank_adjoin_le_one [FiniteDimensional F E]
    (h : ∀ x : E, finrank F F⟮x⟯ ≤ 1) : Subsingleton (IntermediateField F E) :=
  subsingleton_of_bot_eq_top (bot_eq_top_of_finrank_adjoin_le_one h)


theorem minpoly_gen (α : E) :
    minpoly F (AdjoinSimple.gen F α) = minpoly F α := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    ⊢ Eq (minpoly F (IntermediateField.AdjoinSimple.gen F α)) (minpoly F α)
  -/
  rw [← minpoly.algebraMap_eq (algebraMap F⟮α⟯ E).injective, AdjoinSimple.algebraMap_gen]
  /-
    🎉 no goals
  -/


theorem aeval_gen_minpoly (α : E) : aeval (AdjoinSimple.gen F α) (minpoly F α) = 0 := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    ⊢ Eq ((Polynomial.aeval (IntermediateField.AdjoinSimple.gen F α)) (minpoly F α …
  -/
  ext
  /-
    case a
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    ⊢ Eq ↑((Polynomial.aeval (IntermediateField.AdjoinSimple.gen F α)) (minpoly F  …
  -/
  convert minpoly.aeval F α
  /-
    case h.e'_2
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    ⊢ Eq (↑((Polynomial.aeval (IntermediateField.AdjoinSimple.gen F α)) (minpoly F …
  -/
  conv in aeval α => rw [← AdjoinSimple.algebraMap_gen F α]
  /-
    case h.e'_2
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    ⊢ Eq (↑((Polynomial.aeval (IntermediateField.AdjoinSimple.gen F α)) (minpoly F …
  -/
  exact (aeval_algebraMap_apply E (AdjoinSimple.gen F α) _).symm
  /-
    🎉 no goals
  -/

-- Porting note: original proof used `Exists.cases_on`.

/-- algebra isomorphism between `AdjoinRoot` and `F⟮α⟯` -/
@[stacks 09G1 "Algebraic case"]
noncomputable def adjoinRootEquivAdjoin (h : IsIntegral F α) :
    AdjoinRoot (minpoly F α) ≃ₐ[F] F⟮α⟯ :=
  AlgEquiv.ofBijective
    (AdjoinRoot.liftHom (minpoly F α) (AdjoinSimple.gen F α) (aeval_gen_minpoly F α))
    (by
      /-
        F : Type u_1
        inst✝⁴ : Field F
        E : Type u_2
        inst✝³ : Field E
        inst✝² : Algebra F E
        α : E
        K : Type u
        inst✝¹ : Field K
        inst✝ : Algebra F K
        h : IsIntegral F α
        ⊢ Function.Bijective ⇑(AdjoinRoot.liftHom (minpoly F α) (IntermediateField.Adj …
      -/
      set f := AdjoinRoot.lift _ _ (aeval_gen_minpoly F α : _)
      /-
        F : Type u_1
        inst✝⁴ : Field F
        E : Type u_2
        inst✝³ : Field E
        inst✝² : Algebra F E
        α : E
        K : Type u
        inst✝¹ : Field K
        inst✝ : Algebra F K
        h : IsIntegral F α
        f : RingHom (AdjoinRoot (minpoly F α)) (Subtype fun x => Membership.mem (Inter …
        ⊢ Function.Bijective ⇑(AdjoinRoot.liftHom (minpoly F α) (IntermediateField.Adj …
      -/
      haveI := Fact.mk (minpoly.irreducible h)
      /-
        F : Type u_1
        inst✝⁴ : Field F
        E : Type u_2
        inst✝³ : Field E
        inst✝² : Algebra F E
        α : E
        K : Type u
        inst✝¹ : Field K
        inst✝ : Algebra F K
        h : IsIntegral F α
        f : RingHom (AdjoinRoot (minpoly F α)) (Subtype fun x => Membership.mem (Inter …
        this : Fact (Irreducible (minpoly F α))
        ⊢ Function.Bijective ⇑(AdjoinRoot.liftHom (minpoly F α) (IntermediateField.Adj …
      -/
      constructor
        /-
          case left
          F : Type u_1
          inst✝⁴ : Field F
          E : Type u_2
          inst✝³ : Field E
          inst✝² : Algebra F E
          α : E
          K : Type u
          inst✝¹ : Field K
          inst✝ : Algebra F K
          h : IsIntegral F α
          f : RingHom (AdjoinRoot (minpoly F α)) (Subtype fun x => Membership.mem (Inter …
          this : Fact (Irreducible (minpoly F α))
          ⊢ Function.Injective ⇑(AdjoinRoot.liftHom (minpoly F α) (IntermediateField.Adj …
        -/
      · exact RingHom.injective f
        /-
          🎉 no goals
        -/
      · suffices F⟮α⟯.toSubfield ≤ RingHom.fieldRange (F⟮α⟯.toSubfield.subtype.comp f) by
          intro x
          obtain ⟨y, hy⟩ := this (Subtype.mem x)
          exact ⟨y, Subtype.ext hy⟩
        /-
          case right
          F : Type u_1
          inst✝⁴ : Field F
          E : Type u_2
          inst✝³ : Field E
          inst✝² : Algebra F E
          α : E
          K : Type u
          inst✝¹ : Field K
          inst✝ : Algebra F K
          h : IsIntegral F α
          f : RingHom (AdjoinRoot (minpoly F α)) (Subtype fun x => Membership.mem (Inter …
          this : Fact (Irreducible (minpoly F α))
          ⊢ LE.le (IntermediateField.adjoin F (Singleton.singleton α)).toSubfield ((Inte …
        -/
        refine Subfield.closure_le.mpr (Set.union_subset (fun x hx => ?_) ?_)
          /-
            case right.refine_1
            F : Type u_1
            inst✝⁴ : Field F
            E : Type u_2
            inst✝³ : Field E
            inst✝² : Algebra F E
            α : E
            K : Type u
            inst✝¹ : Field K
            inst✝ : Algebra F K
            h : IsIntegral F α
            f : RingHom (AdjoinRoot (minpoly F α)) (Subtype fun x => Membership.mem (Inter …
            this : Fact (Irreducible (minpoly F α))
            x : E
            hx : Membership.mem (Set.range ⇑(algebraMap F E)) x
            ⊢ Membership.mem (↑((IntermediateField.adjoin F (Singleton.singleton α)).toSub …
          -/
        · obtain ⟨y, hy⟩ := hx
          /-
            case right.refine_1.intro
            F : Type u_1
            inst✝⁴ : Field F
            E : Type u_2
            inst✝³ : Field E
            inst✝² : Algebra F E
            α : E
            K : Type u
            inst✝¹ : Field K
            inst✝ : Algebra F K
            h : IsIntegral F α
            f : RingHom (AdjoinRoot (minpoly F α)) (Subtype fun x => Membership.mem (Inter …
            this : Fact (Irreducible (minpoly F α))
            x : E
            y : F
            hy : Eq ((algebraMap F E) y) x
            ⊢ Membership.mem (↑((IntermediateField.adjoin F (Singleton.singleton α)).toSub …
          -/
          refine ⟨y, ?_⟩
          -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
          /-
            case right.refine_1.intro
            F : Type u_1
            inst✝⁴ : Field F
            E : Type u_2
            inst✝³ : Field E
            inst✝² : Algebra F E
            α : E
            K : Type u
            inst✝¹ : Field K
            inst✝ : Algebra F K
            h : IsIntegral F α
            f : RingHom (AdjoinRoot (minpoly F α)) (Subtype fun x => Membership.mem (Inter …
            this : Fact (Irreducible (minpoly F α))
            x : E
            y : F
            hy : Eq ((algebraMap F E) y) x
            ⊢ Eq (((IntermediateField.adjoin F (Singleton.singleton α)).toSubfield.subtype …
          -/
          erw [RingHom.comp_apply, AdjoinRoot.lift_of (aeval_gen_minpoly F α)]
          /-
            case right.refine_1.intro
            F : Type u_1
            inst✝⁴ : Field F
            E : Type u_2
            inst✝³ : Field E
            inst✝² : Algebra F E
            α : E
            K : Type u
            inst✝¹ : Field K
            inst✝ : Algebra F K
            h : IsIntegral F α
            f : RingHom (AdjoinRoot (minpoly F α)) (Subtype fun x => Membership.mem (Inter …
            this : Fact (Irreducible (minpoly F α))
            x : E
            y : F
            hy : Eq ((algebraMap F E) y) x
            ⊢ Eq ((IntermediateField.adjoin F (Singleton.singleton α)).toSubfield.subtype  …
          -/
          exact hy
          /-
            🎉 no goals
          -/
          /-
            case right.refine_2
            F : Type u_1
            inst✝⁴ : Field F
            E : Type u_2
            inst✝³ : Field E
            inst✝² : Algebra F E
            α : E
            K : Type u
            inst✝¹ : Field K
            inst✝ : Algebra F K
            h : IsIntegral F α
            f : RingHom (AdjoinRoot (minpoly F α)) (Subtype fun x => Membership.mem (Inter …
            this : Fact (Irreducible (minpoly F α))
            ⊢ HasSubset.Subset (Singleton.singleton α) ↑((IntermediateField.adjoin F (Sing …
          -/
        · refine Set.singleton_subset_iff.mpr ⟨AdjoinRoot.root (minpoly F α), ?_⟩
          -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
          /-
            case right.refine_2
            F : Type u_1
            inst✝⁴ : Field F
            E : Type u_2
            inst✝³ : Field E
            inst✝² : Algebra F E
            α : E
            K : Type u
            inst✝¹ : Field K
            inst✝ : Algebra F K
            h : IsIntegral F α
            f : RingHom (AdjoinRoot (minpoly F α)) (Subtype fun x => Membership.mem (Inter …
            this : Fact (Irreducible (minpoly F α))
            ⊢ Eq (((IntermediateField.adjoin F (Singleton.singleton α)).toSubfield.subtype …
          -/
          erw [RingHom.comp_apply, AdjoinRoot.lift_root (aeval_gen_minpoly F α)]
          /-
            case right.refine_2
            F : Type u_1
            inst✝⁴ : Field F
            E : Type u_2
            inst✝³ : Field E
            inst✝² : Algebra F E
            α : E
            K : Type u
            inst✝¹ : Field K
            inst✝ : Algebra F K
            h : IsIntegral F α
            f : RingHom (AdjoinRoot (minpoly F α)) (Subtype fun x => Membership.mem (Inter …
            this : Fact (Irreducible (minpoly F α))
            ⊢ Eq ((IntermediateField.adjoin F (Singleton.singleton α)).toSubfield.subtype  …
          -/
          rfl)
          /-
            🎉 no goals
          -/


theorem adjoinRootEquivAdjoin_apply_root (h : IsIntegral F α) :
    adjoinRootEquivAdjoin F h (AdjoinRoot.root (minpoly F α)) = AdjoinSimple.gen F α :=
  AdjoinRoot.lift_root (aeval_gen_minpoly F α)


@[simp]
theorem adjoinRootEquivAdjoin_symm_apply_gen (h : IsIntegral F α) :
    (adjoinRootEquivAdjoin F h).symm (AdjoinSimple.gen F α) = AdjoinRoot.root (minpoly F α) := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    α : E
    h : IsIntegral F α
    ⊢ Eq ((IntermediateField.adjoinRootEquivAdjoin F h).symm (IntermediateField.Ad …
  -/
  rw [AlgEquiv.symm_apply_eq, adjoinRootEquivAdjoin_apply_root]
  /-
    🎉 no goals
  -/


theorem adjoin_root_eq_top (p : K[X]) [Fact (Irreducible p)] : K⟮AdjoinRoot.root p⟯ = ⊤ :=
  (eq_adjoin_of_eq_algebra_adjoin K _ ⊤ (AdjoinRoot.adjoinRoot_eq_top (f := p)).symm).symm


/-- The elements `1, x, ..., x ^ (d - 1)` form a basis for `K⟮x⟯`,
where `d` is the degree of the minimal polynomial of `x`. -/
noncomputable def powerBasisAux {x : L} (hx : IsIntegral K x) :
    Basis (Fin (minpoly K x).natDegree) K K⟮x⟯ :=
  (AdjoinRoot.powerBasis (minpoly.ne_zero hx)).basis.map (adjoinRootEquivAdjoin K hx).toLinearEquiv


/-- The power basis `1, x, ..., x ^ (d - 1)` for `K⟮x⟯`,
where `d` is the degree of the minimal polynomial of `x`. -/
@[simps]
noncomputable def adjoin.powerBasis {x : L} (hx : IsIntegral K x) : PowerBasis K K⟮x⟯ where
  gen := AdjoinSimple.gen K x
  dim := (minpoly K x).natDegree
  basis := powerBasisAux hx
  basis_eq_pow i := by
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    erw [powerBasisAux, Basis.map_apply, PowerBasis.basis_eq_pow, AlgEquiv.toLinearEquiv_apply,
      map_pow, AdjoinRoot.powerBasis_gen, adjoinRootEquivAdjoin_apply_root]


theorem adjoin.finiteDimensional {x : L} (hx : IsIntegral K x) : FiniteDimensional K K⟮x⟯ :=
  (adjoin.powerBasis hx).finite


theorem isAlgebraic_adjoin_simple {x : L} (hx : IsIntegral K x) : Algebra.IsAlgebraic K K⟮x⟯ :=
  have := adjoin.finiteDimensional hx; Algebra.IsAlgebraic.of_finite K K⟮x⟯


/-- If `x` is an algebraic element of field `K`, then its minimal polynomial has degree
`[K(x) : K]`. -/
@[stacks 09GN]
theorem adjoin.finrank {x : L} (hx : IsIntegral K x) :
    Module.finrank K K⟮x⟯ = (minpoly K x).natDegree := by
  /-
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    hx : IsIntegral K x
    ⊢ Eq (Module.finrank K (Subtype fun x_1 => Membership.mem (IntermediateField.a …
  -/
  rw [PowerBasis.finrank (adjoin.powerBasis hx : _)]
  /-
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    hx : IsIntegral K x
    ⊢ Eq (IntermediateField.adjoin.powerBasis hx).dim (minpoly K x).natDegree
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `K / E / F` is a field extension tower, `S ⊂ K` is such that `F(S) = K`,
then `E(S) = K`. -/
theorem adjoin_eq_top_of_adjoin_eq_top [Algebra E K] [IsScalarTower F E K]
    {S : Set K} (hprim : adjoin F S = ⊤) : adjoin E S = ⊤ :=
  restrictScalars_injective F <| by
    rw [restrictScalars_top, ← top_le_iff, ← hprim, adjoin_le_iff,
      coe_restrictScalars, ← adjoin_le_iff]


/-- If `E / F` is a finite extension such that `E = F(α)`, then for any intermediate field `K`, the
`F` adjoin the coefficients of `minpoly K α` is equal to `K` itself. -/
theorem adjoin_minpoly_coeff_of_exists_primitive_element
    [FiniteDimensional F E] (hprim : adjoin F {α} = ⊤) (K : IntermediateField F E) :
    adjoin F ((minpoly K α).map (algebraMap K E)).coeffs = K := by
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    α : E
    inst✝ : FiniteDimensional F E
    hprim : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    K : IntermediateField F E
    ⊢ Eq (IntermediateField.adjoin F ↑(Polynomial.map (algebraMap (Subtype fun x = …
  -/
  set g := (minpoly K α).map (algebraMap K E)
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    α : E
    inst✝ : FiniteDimensional F E
    hprim : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    K : IntermediateField F E
    g : Polynomial E := Polynomial.map (algebraMap (Subtype fun x => Membership.me …
    ⊢ Eq (IntermediateField.adjoin F ↑g.coeffs) K
  -/
  set K' : IntermediateField F E := adjoin F g.coeffs
  have hsub : K' ≤ K := by
    refine adjoin_le_iff.mpr fun x ↦ ?_
    rw [Finset.mem_coe, mem_coeffs_iff]
    rintro ⟨n, -, rfl⟩
    rw [coeff_map]
    apply Subtype.mem
  have dvd_g : minpoly K' α ∣ g.toSubring K'.toSubring (subset_adjoin F _) := by
    apply minpoly.dvd
    erw [aeval_def, eval₂_eq_eval_map, g.map_toSubring K'.toSubring, eval_map, ← aeval_def]
    exact minpoly.aeval K α
  have finrank_eq : ∀ K : IntermediateField F E, finrank K E = natDegree (minpoly K α) := by
    intro K
    have := adjoin.finrank (.of_finite K α)
    erw [adjoin_eq_top_of_adjoin_eq_top F hprim, finrank_top K E] at this
    exact this
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    α : E
    inst✝ : FiniteDimensional F E
    hprim : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    K : IntermediateField F E
    g : Polynomial E := Polynomial.map (algebraMap (Subtype fun x => Membership.me …
    K' : IntermediateField F E := IntermediateField.adjoin F ↑g.coeffs
    hsub : LE.le K' K
    dvd_g : Dvd.dvd (minpoly (Subtype fun x => Membership.mem K' x) α) (g.toSubrin …
    finrank_eq : ∀ (K : IntermediateField F E), Eq (Module.finrank (Subtype fun x  …
    ⊢ Eq K' K
  -/
  refine eq_of_le_of_finrank_le' hsub ?_
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    α : E
    inst✝ : FiniteDimensional F E
    hprim : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    K : IntermediateField F E
    g : Polynomial E := Polynomial.map (algebraMap (Subtype fun x => Membership.me …
    K' : IntermediateField F E := IntermediateField.adjoin F ↑g.coeffs
    hsub : LE.le K' K
    dvd_g : Dvd.dvd (minpoly (Subtype fun x => Membership.mem K' x) α) (g.toSubrin …
    finrank_eq : ∀ (K : IntermediateField F E), Eq (Module.finrank (Subtype fun x  …
    ⊢ LE.le (Module.finrank (Subtype fun x => Membership.mem K' x) E) (Module.finr …
  -/
  simp_rw [finrank_eq]
  convert natDegree_le_of_dvd dvd_g
    ((g.monic_toSubring _ _).mpr <| (minpoly.monic <| .of_finite K α).map _).ne_zero using 1
  /-
    case h.e'_4
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    α : E
    inst✝ : FiniteDimensional F E
    hprim : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    K : IntermediateField F E
    g : Polynomial E := Polynomial.map (algebraMap (Subtype fun x => Membership.me …
    K' : IntermediateField F E := IntermediateField.adjoin F ↑g.coeffs
    hsub : LE.le K' K
    dvd_g : Dvd.dvd (minpoly (Subtype fun x => Membership.mem K' x) α) (g.toSubrin …
    finrank_eq : ∀ (K : IntermediateField F E), Eq (Module.finrank (Subtype fun x  …
    ⊢ Eq (minpoly (Subtype fun x => Membership.mem K x) α).natDegree (g.toSubring  …
  -/
  rw [natDegree_toSubring, natDegree_map]
  /-
    🎉 no goals
  -/


instance : Module.Finite F (⊥ : IntermediateField F E) := Subalgebra.finite_bot


variable {F} in
/-- If `E / F` is an infinite algebraic extension, then there exists an intermediate field
`L / F` with arbitrarily large finite extension degree. -/
theorem exists_lt_finrank_of_infinite_dimensional
    [Algebra.IsAlgebraic F E] (hnfd : ¬ FiniteDimensional F E) (n : ℕ) :
    ∃ L : IntermediateField F E, FiniteDimensional F L ∧ n < finrank F L := by
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    hnfd : Not (FiniteDimensional F E)
    n : Nat
    ⊢ Exists fun L => And (FiniteDimensional F (Subtype fun x => Membership.mem L  …
  -/
  induction' n with n ih
    /-
      case zero
      F : Type u_1
      inst✝³ : Field F
      E : Type u_2
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Algebra.IsAlgebraic F E
      hnfd : Not (FiniteDimensional F E)
      ⊢ Exists fun L => And (FiniteDimensional F (Subtype fun x => Membership.mem L  …
    -/
  · exact ⟨⊥, Subalgebra.finite_bot, finrank_pos⟩
    /-
      🎉 no goals
    -/
  /-
    case succ
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    hnfd : Not (FiniteDimensional F E)
    n : Nat
    ih : Exists fun L => And (FiniteDimensional F (Subtype fun x => Membership.mem …
    ⊢ Exists fun L => And (FiniteDimensional F (Subtype fun x => Membership.mem L  …
  -/
  obtain ⟨L, fin, hn⟩ := ih
  obtain ⟨x, hx⟩ : ∃ x : E, x ∉ L := by
    contrapose! hnfd
    rw [show L = ⊤ from eq_top_iff.2 fun x _ ↦ hnfd x] at fin
    exact topEquiv.toLinearEquiv.finiteDimensional
  /-
    case succ.intro.intro.intro
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    hnfd : Not (FiniteDimensional F E)
    n : Nat
    L : IntermediateField F E
    fin : FiniteDimensional F (Subtype fun x => Membership.mem L x)
    hn : LT.lt n (Module.finrank F (Subtype fun x => Membership.mem L x))
    x : E
    hx : Not (Membership.mem L x)
    ⊢ Exists fun L => And (FiniteDimensional F (Subtype fun x => Membership.mem L  …
  -/
  let L' := L ⊔ F⟮x⟯
  /-
    case succ.intro.intro.intro
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    hnfd : Not (FiniteDimensional F E)
    n : Nat
    L : IntermediateField F E
    fin : FiniteDimensional F (Subtype fun x => Membership.mem L x)
    hn : LT.lt n (Module.finrank F (Subtype fun x => Membership.mem L x))
    x : E
    hx : Not (Membership.mem L x)
    L' : IntermediateField F E := Max.max L (IntermediateField.adjoin F (Singleton …
    ⊢ Exists fun L => And (FiniteDimensional F (Subtype fun x => Membership.mem L  …
  -/
  haveI := adjoin.finiteDimensional (Algebra.IsIntegral.isIntegral (R := F) x)
  /-
    case succ.intro.intro.intro
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    hnfd : Not (FiniteDimensional F E)
    n : Nat
    L : IntermediateField F E
    fin : FiniteDimensional F (Subtype fun x => Membership.mem L x)
    hn : LT.lt n (Module.finrank F (Subtype fun x => Membership.mem L x))
    x : E
    hx : Not (Membership.mem L x)
    L' : IntermediateField F E := Max.max L (IntermediateField.adjoin F (Singleton …
    this : FiniteDimensional F (Subtype fun x_1 => Membership.mem (IntermediateFie …
    ⊢ Exists fun L => And (FiniteDimensional F (Subtype fun x => Membership.mem L  …
  -/
  refine ⟨L', inferInstance, by_contra fun h ↦ ?_⟩
  /-
    case succ.intro.intro.intro
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    hnfd : Not (FiniteDimensional F E)
    n : Nat
    L : IntermediateField F E
    fin : FiniteDimensional F (Subtype fun x => Membership.mem L x)
    hn : LT.lt n (Module.finrank F (Subtype fun x => Membership.mem L x))
    x : E
    hx : Not (Membership.mem L x)
    L' : IntermediateField F E := Max.max L (IntermediateField.adjoin F (Singleton …
    this : FiniteDimensional F (Subtype fun x_1 => Membership.mem (IntermediateFie …
    h : Not (LT.lt (HAdd.hAdd n 1) (Module.finrank F (Subtype fun x => Membership. …
    ⊢ False
  -/
  have h1 : L = L' := eq_of_le_of_finrank_le le_sup_left ((not_lt.1 h).trans hn)
  /-
    case succ.intro.intro.intro
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    hnfd : Not (FiniteDimensional F E)
    n : Nat
    L : IntermediateField F E
    fin : FiniteDimensional F (Subtype fun x => Membership.mem L x)
    hn : LT.lt n (Module.finrank F (Subtype fun x => Membership.mem L x))
    x : E
    hx : Not (Membership.mem L x)
    L' : IntermediateField F E := Max.max L (IntermediateField.adjoin F (Singleton …
    this : FiniteDimensional F (Subtype fun x_1 => Membership.mem (IntermediateFie …
    h : Not (LT.lt (HAdd.hAdd n 1) (Module.finrank F (Subtype fun x => Membership. …
    h1 : Eq L L'
    ⊢ False
  -/
  have h2 : F⟮x⟯ ≤ L' := le_sup_right
  /-
    case succ.intro.intro.intro
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    hnfd : Not (FiniteDimensional F E)
    n : Nat
    L : IntermediateField F E
    fin : FiniteDimensional F (Subtype fun x => Membership.mem L x)
    hn : LT.lt n (Module.finrank F (Subtype fun x => Membership.mem L x))
    x : E
    hx : Not (Membership.mem L x)
    L' : IntermediateField F E := Max.max L (IntermediateField.adjoin F (Singleton …
    this : FiniteDimensional F (Subtype fun x_1 => Membership.mem (IntermediateFie …
    h : Not (LT.lt (HAdd.hAdd n 1) (Module.finrank F (Subtype fun x => Membership. …
    h1 : Eq L L'
    h2 : LE.le (IntermediateField.adjoin F (Singleton.singleton x)) L'
    ⊢ False
  -/
  exact hx <| (h1.symm ▸ h2) <| mem_adjoin_simple_self F x
  /-
    🎉 no goals
  -/


theorem _root_.minpoly.natDegree_le (x : L) [FiniteDimensional K L] :
    (minpoly K x).natDegree ≤ finrank K L :=
  le_of_eq_of_le (IntermediateField.adjoin.finrank (.of_finite _ _)).symm
    K⟮x⟯.toSubmodule.finrank_le


theorem _root_.minpoly.degree_le (x : L) [FiniteDimensional K L] :
    (minpoly K x).degree ≤ finrank K L :=
  degree_le_of_natDegree_le (minpoly.natDegree_le x)


/-- If `x : L` is an integral element in a field extension `L` over `K`, then the degree of the
  minimal polynomial of `x` over `K` divides `[L : K]`.-/
theorem _root_.minpoly.degree_dvd {x : L} (hx : IsIntegral K x) :
    (minpoly K x).natDegree ∣ finrank K L := by
  /-
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    hx : IsIntegral K x
    ⊢ Dvd.dvd (minpoly K x).natDegree (Module.finrank K L)
  -/
  rw [dvd_iff_exists_eq_mul_left, ← IntermediateField.adjoin.finrank hx]
  /-
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    hx : IsIntegral K x
    ⊢ Exists fun c => Eq (Module.finrank K L) (HMul.hMul c (Module.finrank K (Subt …
  -/
  use finrank K⟮x⟯ L
  /-
    case h
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    hx : IsIntegral K x
    ⊢ Eq (Module.finrank K L) (HMul.hMul (Module.finrank (Subtype fun x_1 => Membe …
  -/
  rw [mul_comm, finrank_mul_finrank]
  /-
    🎉 no goals
  -/

-- TODO: generalize to `Sort`

/-- A compositum of algebraic extensions is algebraic -/
theorem isAlgebraic_iSup {ι : Type*} {t : ι → IntermediateField K L}
    (h : ∀ i, Algebra.IsAlgebraic K (t i)) :
    Algebra.IsAlgebraic K (⨆ i, t i : IntermediateField K L) := by
  /-
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ι : Type u_4
    t : ι → IntermediateField K L
    h : ∀ (i : ι), Algebra.IsAlgebraic K (Subtype fun x => Membership.mem (t i) x)
    ⊢ Algebra.IsAlgebraic K (Subtype fun x => Membership.mem (iSup fun i => t i) x)
  -/
  constructor
  /-
    case isAlgebraic
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ι : Type u_4
    t : ι → IntermediateField K L
    h : ∀ (i : ι), Algebra.IsAlgebraic K (Subtype fun x => Membership.mem (t i) x)
    ⊢ ∀ (x : Subtype fun x => Membership.mem (iSup fun i => t i) x), IsAlgebraic K x
  -/
  rintro ⟨x, hx⟩
  /-
    case isAlgebraic.mk
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ι : Type u_4
    t : ι → IntermediateField K L
    h : ∀ (i : ι), Algebra.IsAlgebraic K (Subtype fun x => Membership.mem (t i) x)
    x : L
    hx : Membership.mem (iSup fun i => t i) x
    ⊢ IsAlgebraic K ⟨x, hx⟩
  -/
  obtain ⟨s, hx⟩ := exists_finset_of_mem_supr' hx
  /-
    case isAlgebraic.mk.intro
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ι : Type u_4
    t : ι → IntermediateField K L
    h : ∀ (i : ι), Algebra.IsAlgebraic K (Subtype fun x => Membership.mem (t i) x)
    x : L
    hx✝ : Membership.mem (iSup fun i => t i) x
    s : Finset (Sigma fun i => Subtype fun x => Membership.mem (t i) x)
    hx : Membership.mem (iSup fun i => iSup fun h => IntermediateField.adjoin K (S …
    ⊢ IsAlgebraic K ⟨x, hx✝⟩
  -/
  rw [isAlgebraic_iff, Subtype.coe_mk, ← Subtype.coe_mk (p := (· ∈ _)) x hx, ← isAlgebraic_iff]
  haveI : ∀ i : Σ i, t i, FiniteDimensional K K⟮(i.2 : L)⟯ := fun ⟨i, x⟩ ↦
    adjoin.finiteDimensional (isIntegral_iff.1 (Algebra.IsIntegral.isIntegral x))
  /-
    case isAlgebraic.mk.intro
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ι : Type u_4
    t : ι → IntermediateField K L
    h : ∀ (i : ι), Algebra.IsAlgebraic K (Subtype fun x => Membership.mem (t i) x)
    x : L
    hx✝ : Membership.mem (iSup fun i => t i) x
    s : Finset (Sigma fun i => Subtype fun x => Membership.mem (t i) x)
    hx : Membership.mem (iSup fun i => iSup fun h => IntermediateField.adjoin K (S …
    this : ∀ (i : Sigma fun i => Subtype fun x => Membership.mem (t i) x), FiniteD …
    ⊢ IsAlgebraic K ⟨x, hx⟩
  -/
  apply IsAlgebraic.of_finite
  /-
    🎉 no goals
  -/


theorem isAlgebraic_adjoin {S : Set L} (hS : ∀ x ∈ S, IsIntegral K x) :
    Algebra.IsAlgebraic K (adjoin K S) := by
  /-
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    S : Set L
    hS : ∀ (x : L), Membership.mem S x → IsIntegral K x
    ⊢ Algebra.IsAlgebraic K (Subtype fun x => Membership.mem (IntermediateField.ad …
  -/
  rw [← biSup_adjoin_simple, ← iSup_subtype'']
  /-
    K : Type u
    inst✝² : Field K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra K L
    S : Set L
    hS : ∀ (x : L), Membership.mem S x → IsIntegral K x
    ⊢ Algebra.IsAlgebraic K (Subtype fun x => Membership.mem (iSup fun i => Interm …
  -/
  exact isAlgebraic_iSup fun x ↦ isAlgebraic_adjoin_simple (hS x x.2)
  /-
    🎉 no goals
  -/


/-- If `L / K` is a field extension, `S` is a finite subset of `L`, such that every element of `S`
is integral (= algebraic) over `K`, then `K(S) / K` is a finite extension.
A direct corollary of `finiteDimensional_iSup_of_finite`. -/
theorem finiteDimensional_adjoin {S : Set L} [Finite S] (hS : ∀ x ∈ S, IsIntegral K x) :
    FiniteDimensional K (adjoin K S) := by
  /-
    K : Type u
    inst✝³ : Field K
    L : Type u_3
    inst✝² : Field L
    inst✝¹ : Algebra K L
    S : Set L
    inst✝ : Finite ↑S
    hS : ∀ (x : L), Membership.mem S x → IsIntegral K x
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (IntermediateField.adjo …
  -/
  rw [← biSup_adjoin_simple, ← iSup_subtype'']
  /-
    K : Type u
    inst✝³ : Field K
    L : Type u_3
    inst✝² : Field L
    inst✝¹ : Algebra K L
    S : Set L
    inst✝ : Finite ↑S
    hS : ∀ (x : L), Membership.mem S x → IsIntegral K x
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (iSup fun i => Intermed …
  -/
  haveI (x : S) := adjoin.finiteDimensional (hS x.1 x.2)
  /-
    K : Type u
    inst✝³ : Field K
    L : Type u_3
    inst✝² : Field L
    inst✝¹ : Algebra K L
    S : Set L
    inst✝ : Finite ↑S
    hS : ∀ (x : L), Membership.mem S x → IsIntegral K x
    this : ∀ (x : ↑S), FiniteDimensional K (Subtype fun x_1 => Membership.mem (Int …
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (iSup fun i => Intermed …
  -/
  exact finiteDimensional_iSup_of_finite
  /-
    🎉 no goals
  -/


/-- Algebra homomorphism `F⟮α⟯ →ₐ[F] K` are in bijection with the set of roots
of `minpoly α` in `K`. -/
noncomputable def algHomAdjoinIntegralEquiv (h : IsIntegral F α) :
    (F⟮α⟯ →ₐ[F] K) ≃ { x // x ∈ (minpoly F α).aroots K } :=
  (adjoin.powerBasis h).liftEquiv'.trans
    ((Equiv.refl _).subtypeEquiv fun x => by
      /-
        F : Type u_1
        inst✝⁴ : Field F
        E : Type u_2
        inst✝³ : Field E
        inst✝² : Algebra F E
        α : E
        K : Type u
        inst✝¹ : Field K
        inst✝ : Algebra F K
        h : IsIntegral F α
        x : K
        ⊢ Iff (Membership.mem ((minpoly F (IntermediateField.adjoin.powerBasis h).gen) …
      -/
      rw [adjoin.powerBasis_gen, minpoly_gen, Equiv.refl_apply])
      /-
        🎉 no goals
      -/


lemma algHomAdjoinIntegralEquiv_symm_apply_gen (h : IsIntegral F α)
    (x : { x // x ∈ (minpoly F α).aroots K }) :
    (algHomAdjoinIntegralEquiv F h).symm x (AdjoinSimple.gen F α) = x :=
  (adjoin.powerBasis h).lift_gen x.val <| by
    /-
      F : Type u_1
      inst✝⁴ : Field F
      E : Type u_2
      inst✝³ : Field E
      inst✝² : Algebra F E
      α : E
      K : Type u
      inst✝¹ : Field K
      inst✝ : Algebra F K
      h : IsIntegral F α
      x : Subtype fun x => Membership.mem ((minpoly F α).aroots K) x
      ⊢ Eq ((Polynomial.aeval ↑x) (minpoly F (IntermediateField.adjoin.powerBasis h) …
    -/
    rw [adjoin.powerBasis_gen, minpoly_gen]; exact (mem_aroots.mp x.2).2
                                             /-
                                               🎉 no goals
                                             -/


/-- Fintype of algebra homomorphism `F⟮α⟯ →ₐ[F] K` -/
noncomputable def fintypeOfAlgHomAdjoinIntegral (h : IsIntegral F α) : Fintype (F⟮α⟯ →ₐ[F] K) :=
  PowerBasis.AlgHom.fintype (adjoin.powerBasis h)


theorem card_algHom_adjoin_integral (h : IsIntegral F α) (h_sep : IsSeparable F α)
    (h_splits : (minpoly F α).Splits (algebraMap F K)) :
    @Fintype.card (F⟮α⟯ →ₐ[F] K) (fintypeOfAlgHomAdjoinIntegral F h) = (minpoly F α).natDegree := by
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    α : E
    K : Type u
    inst✝¹ : Field K
    inst✝ : Algebra F K
    h : IsIntegral F α
    h_sep : IsSeparable F α
    h_splits : Polynomial.Splits (algebraMap F K) (minpoly F α)
    ⊢ Eq (Fintype.card (AlgHom F (Subtype fun x => Membership.mem (IntermediateFie …
  -/
  rw [AlgHom.card_of_powerBasis] <;>
    /-
      F : Type u_1
      inst✝⁴ : Field F
      E : Type u_2
      inst✝³ : Field E
      inst✝² : Algebra F E
      α : E
      K : Type u
      inst✝¹ : Field K
      inst✝ : Algebra F K
      h : IsIntegral F α
      h_sep : IsSeparable F α
      h_splits : Polynomial.Splits (algebraMap F K) (minpoly F α)
      ⊢ Eq (IntermediateField.adjoin.powerBasis h).dim (minpoly F α).natDegree
    -/
    /-
      🎉 no goals
    -/
    simp only [IsSeparable, adjoin.powerBasis_dim, adjoin.powerBasis_gen, minpoly_gen, h_splits]
    /-
      🎉 no goals
    -/
  /-
    case h_sep
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    α : E
    K : Type u
    inst✝¹ : Field K
    inst✝ : Algebra F K
    h : IsIntegral F α
    h_sep : IsSeparable F α
    h_splits : Polynomial.Splits (algebraMap F K) (minpoly F α)
    ⊢ (minpoly F α).Separable
  -/
  exact h_sep
  /-
    🎉 no goals
  -/

-- Apparently `K⟮root f⟯ →+* K⟮root f⟯` is expensive to unify during instance synthesis.

open Module AdjoinRoot in
/-- Let `f, g` be monic polynomials over `K`. If `f` is irreducible, and `g(x) - α` is irreducible
in `K⟮α⟯` with `α` a root of `f`, then `f(g(x))` is irreducible. -/
theorem _root_.Polynomial.irreducible_comp {f g : K[X]} (hfm : f.Monic) (hgm : g.Monic)
    (hf : Irreducible f)
    (hg : ∀ (E : Type u) [Field E] [Algebra K E] (x : E) (_ : minpoly K x = f),
      Irreducible (g.map (algebraMap _ _) - C (AdjoinSimple.gen K x))) :
    Irreducible (f.comp g) := by
  have hf' : natDegree f ≠ 0 :=
    fun e ↦ not_irreducible_C (f.coeff 0) (eq_C_of_natDegree_eq_zero e ▸ hf)
  have hg' : natDegree g ≠ 0 := by
    have := Fact.mk hf
    intro e
    apply not_irreducible_C ((g.map (algebraMap _ _)).coeff 0 - AdjoinSimple.gen K (root f))
    -- Needed to specialize `map_sub` to avoid a timeout https://github.com/leanprover-community/mathlib4/pull/8386
    rw [RingHom.map_sub, coeff_map, ← map_C, ← eq_C_of_natDegree_eq_zero e]
    apply hg (AdjoinRoot f)
    rw [AdjoinRoot.minpoly_root hf.ne_zero, hfm, inv_one, map_one, mul_one]
  /-
    K : Type u
    inst✝ : Field K
    f g : Polynomial K
    hfm : f.Monic
    hgm : g.Monic
    hf : Irreducible f
    hg : ∀ (E : Type u) [inst : Field E] [inst_1 : Algebra K E] (x : E), Eq (minpo …
    hf' : Ne f.natDegree 0
    hg' : Ne g.natDegree 0
    ⊢ Irreducible (f.comp g)
  -/
  have H₁ : f.comp g ≠ 0 := fun h ↦ by simpa [hf', hg', natDegree_comp] using congr_arg natDegree h
  have H₂ : ¬ IsUnit (f.comp g) := fun h ↦
    by simpa [hf', hg', natDegree_comp] using natDegree_eq_zero_of_isUnit h
  /-
    K : Type u
    inst✝ : Field K
    f g : Polynomial K
    hfm : f.Monic
    hgm : g.Monic
    hf : Irreducible f
    hg : ∀ (E : Type u) [inst : Field E] [inst_1 : Algebra K E] (x : E), Eq (minpo …
    hf' : Ne f.natDegree 0
    hg' : Ne g.natDegree 0
    H₁ : Ne (f.comp g) 0
    H₂ : Not (IsUnit (f.comp g))
    ⊢ Irreducible (f.comp g)
  -/
  have ⟨p, hp₁, hp₂⟩ := WfDvdMonoid.exists_irreducible_factor H₂ H₁
  suffices natDegree p = natDegree f * natDegree g from (associated_of_dvd_of_natDegree_le hp₂ H₁
    (this.trans natDegree_comp.symm).ge).irreducible hp₁
  /-
    K : Type u
    inst✝ : Field K
    f g : Polynomial K
    hfm : f.Monic
    hgm : g.Monic
    hf : Irreducible f
    hg : ∀ (E : Type u) [inst : Field E] [inst_1 : Algebra K E] (x : E), Eq (minpo …
    hf' : Ne f.natDegree 0
    hg' : Ne g.natDegree 0
    H₁ : Ne (f.comp g) 0
    H₂ : Not (IsUnit (f.comp g))
    p : Polynomial K
    hp₁ : Irreducible p
    hp₂ : Dvd.dvd p (f.comp g)
    ⊢ Eq p.natDegree (HMul.hMul f.natDegree g.natDegree)
  -/
  have := Fact.mk hp₁
  /-
    K : Type u
    inst✝ : Field K
    f g : Polynomial K
    hfm : f.Monic
    hgm : g.Monic
    hf : Irreducible f
    hg : ∀ (E : Type u) [inst : Field E] [inst_1 : Algebra K E] (x : E), Eq (minpo …
    hf' : Ne f.natDegree 0
    hg' : Ne g.natDegree 0
    H₁ : Ne (f.comp g) 0
    H₂ : Not (IsUnit (f.comp g))
    p : Polynomial K
    hp₁ : Irreducible p
    hp₂ : Dvd.dvd p (f.comp g)
    this : Fact (Irreducible p)
    ⊢ Eq p.natDegree (HMul.hMul f.natDegree g.natDegree)
  -/
  let Kx := AdjoinRoot p
  /-
    K : Type u
    inst✝ : Field K
    f g : Polynomial K
    hfm : f.Monic
    hgm : g.Monic
    hf : Irreducible f
    hg : ∀ (E : Type u) [inst : Field E] [inst_1 : Algebra K E] (x : E), Eq (minpo …
    hf' : Ne f.natDegree 0
    hg' : Ne g.natDegree 0
    H₁ : Ne (f.comp g) 0
    H₂ : Not (IsUnit (f.comp g))
    p : Polynomial K
    hp₁ : Irreducible p
    hp₂ : Dvd.dvd p (f.comp g)
    this : Fact (Irreducible p)
    Kx : Type u := AdjoinRoot p
    ⊢ Eq p.natDegree (HMul.hMul f.natDegree g.natDegree)
  -/
  letI := (AdjoinRoot.powerBasis hp₁.ne_zero).finite
  have key₁ : f = minpoly K (aeval (root p) g) := by
    refine minpoly.eq_of_irreducible_of_monic hf ?_ hfm
    rw [← aeval_comp]
    exact aeval_eq_zero_of_dvd_aeval_eq_zero hp₂ (AdjoinRoot.eval₂_root p)
  have key₁' : finrank K K⟮aeval (root p) g⟯ = natDegree f := by
    rw [adjoin.finrank, ← key₁]
    exact IsIntegral.of_finite _ _
  have key₂ : g.map (algebraMap _ _) - C (AdjoinSimple.gen K (aeval (root p) g)) =
      minpoly K⟮aeval (root p) g⟯ (root p) :=
    minpoly.eq_of_irreducible_of_monic (hg _ _ key₁.symm) (by simp [AdjoinSimple.gen])
      (Monic.sub_of_left (hgm.map _) (degree_lt_degree (by simpa [Nat.pos_iff_ne_zero] using hg')))
  have key₂' : finrank K⟮aeval (root p) g⟯ Kx = natDegree g := by
    trans natDegree (minpoly K⟮aeval (root p) g⟯ (root p))
    · have : K⟮aeval (root p) g⟯⟮root p⟯ = ⊤ := by
        apply restrictScalars_injective K
        rw [restrictScalars_top, adjoin_adjoin_left, Set.union_comm, ← adjoin_adjoin_left,
          adjoin_root_eq_top p, restrictScalars_adjoin]
        simp
      rw [← finrank_top', ← this, adjoin.finrank]
      exact IsIntegral.of_finite _ _
    · simp [← key₂]
  /-
    K : Type u
    inst✝ : Field K
    f g : Polynomial K
    hfm : f.Monic
    hgm : g.Monic
    hf : Irreducible f
    hg : ∀ (E : Type u) [inst : Field E] [inst_1 : Algebra K E] (x : E), Eq (minpo …
    hf' : Ne f.natDegree 0
    hg' : Ne g.natDegree 0
    H₁ : Ne (f.comp g) 0
    H₂ : Not (IsUnit (f.comp g))
    p : Polynomial K
    hp₁ : Irreducible p
    hp₂ : Dvd.dvd p (f.comp g)
    this✝ : Fact (Irreducible p)
    Kx : Type u := AdjoinRoot p
    this : Module.Finite K (AdjoinRoot p) := PowerBasis.finite (AdjoinRoot.powerBa …
    key₁ : Eq f (minpoly K ((Polynomial.aeval (AdjoinRoot.root p)) g))
    key₁' : Eq (Module.finrank K (Subtype fun x => Membership.mem (IntermediateFie …
    key₂ : Eq (HSub.hSub (Polynomial.map (algebraMap K (Subtype fun x => Membershi …
    key₂' : Eq (Module.finrank (Subtype fun x => Membership.mem (IntermediateField …
    ⊢ Eq p.natDegree (HMul.hMul f.natDegree g.natDegree)
  -/
  have := Module.finrank_mul_finrank K K⟮aeval (root p) g⟯ Kx
  /-
    K : Type u
    inst✝ : Field K
    f g : Polynomial K
    hfm : f.Monic
    hgm : g.Monic
    hf : Irreducible f
    hg : ∀ (E : Type u) [inst : Field E] [inst_1 : Algebra K E] (x : E), Eq (minpo …
    hf' : Ne f.natDegree 0
    hg' : Ne g.natDegree 0
    H₁ : Ne (f.comp g) 0
    H₂ : Not (IsUnit (f.comp g))
    p : Polynomial K
    hp₁ : Irreducible p
    hp₂ : Dvd.dvd p (f.comp g)
    this✝¹ : Fact (Irreducible p)
    Kx : Type u := AdjoinRoot p
    this✝ : Module.Finite K (AdjoinRoot p) := PowerBasis.finite (AdjoinRoot.powerB …
    key₁ : Eq f (minpoly K ((Polynomial.aeval (AdjoinRoot.root p)) g))
    key₁' : Eq (Module.finrank K (Subtype fun x => Membership.mem (IntermediateFie …
    key₂ : Eq (HSub.hSub (Polynomial.map (algebraMap K (Subtype fun x => Membershi …
    key₂' : Eq (Module.finrank (Subtype fun x => Membership.mem (IntermediateField …
    this : Eq (HMul.hMul (Module.finrank K (Subtype fun x => Membership.mem (Inter …
    ⊢ Eq p.natDegree (HMul.hMul f.natDegree g.natDegree)
  -/
  rwa [key₁', key₂', (AdjoinRoot.powerBasis hp₁.ne_zero).finrank, powerBasis_dim, eq_comm] at this
  /-
    🎉 no goals
  -/


/-- An intermediate field `S` is finitely generated if there exists `t : Finset E` such that
`IntermediateField.adjoin F t = S`. -/
@[stacks 09FZ "second part"]
def FG (S : IntermediateField F E) : Prop :=
  ∃ t : Finset E, adjoin F ↑t = S


theorem fg_adjoin_finset (t : Finset E) : (adjoin F (↑t : Set E)).FG :=
  ⟨t, rfl⟩


theorem fg_def {S : IntermediateField F E} : S.FG ↔ ∃ t : Set E, Set.Finite t ∧ adjoin F t = S :=
  Iff.symm Set.exists_finite_iff_finset


theorem fg_adjoin_of_finite {t : Set E} (h : Set.Finite t) : (adjoin F t).FG :=
  fg_def.mpr ⟨t, h, rfl⟩


theorem fg_bot : (⊥ : IntermediateField F E).FG :=
  -- Porting note: was `⟨∅, adjoin_empty F E⟩`
         /-
           F : Type u_1
           inst✝² : Field F
           E : Type u_2
           inst✝¹ : Field E
           inst✝ : Algebra F E
           ⊢ Eq (IntermediateField.adjoin F ↑EmptyCollection.emptyCollection) Bot.bot
         -/
  ⟨∅, by simp only [Finset.coe_empty, adjoin_empty]⟩
         /-
           🎉 no goals
         -/


theorem fg_sup {S T : IntermediateField F E} (hS : S.FG) (hT : T.FG) : (S ⊔ T).FG := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S T : IntermediateField F E
    hS : S.FG
    hT : T.FG
    ⊢ (Max.max S T).FG
  -/
  obtain ⟨s, rfl⟩ := hS; obtain ⟨t, rfl⟩ := hT
  /-
    case intro.intro
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    s t : Finset E
    ⊢ (Max.max (IntermediateField.adjoin F ↑s) (IntermediateField.adjoin F ↑t)).FG
  -/
  classical rw [← adjoin_union, ← Finset.coe_union]
  /-
    case intro.intro
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    s t : Finset E
    ⊢ (IntermediateField.adjoin F ↑(Union.union s t)).FG
  -/
  exact fg_adjoin_finset _
  /-
    🎉 no goals
  -/


theorem fg_iSup {ι : Sort*} [Finite ι] {S : ι → IntermediateField F E} (h : ∀ i, (S i).FG) :
    (⨆ i, S i).FG := by
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    ι : Sort u_3
    inst✝ : Finite ι
    S : ι → IntermediateField F E
    h : ∀ (i : ι), (S i).FG
    ⊢ (iSup fun i => S i).FG
  -/
  choose s hs using h
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    ι : Sort u_3
    inst✝ : Finite ι
    S : ι → IntermediateField F E
    s : ι → Finset E
    hs : ∀ (i : ι), Eq (IntermediateField.adjoin F ↑(s i)) (S i)
    ⊢ (iSup fun i => S i).FG
  -/
  simp_rw [← hs, ← adjoin_iUnion]
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    ι : Sort u_3
    inst✝ : Finite ι
    S : ι → IntermediateField F E
    s : ι → Finset E
    hs : ∀ (i : ι), Eq (IntermediateField.adjoin F ↑(s i)) (S i)
    ⊢ (IntermediateField.adjoin F (Set.iUnion fun i => ↑(s i))).FG
  -/
  exact fg_adjoin_of_finite (Set.finite_iUnion fun _ ↦ Finset.finite_toSet _)
  /-
    🎉 no goals
  -/


theorem fg_of_fg_toSubalgebra (S : IntermediateField F E) (h : S.toSubalgebra.FG) : S.FG := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : IntermediateField F E
    h : S.FG
    ⊢ S.FG
  -/
  cases' h with t ht
  /-
    case intro
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    S : IntermediateField F E
    t : Finset E
    ht : Eq (Algebra.adjoin F ↑t) S.toSubalgebra
    ⊢ S.FG
  -/
  exact ⟨t, (eq_adjoin_of_eq_algebra_adjoin _ _ _ ht.symm).symm⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")] alias fG_of_fG_toSubalgebra := fg_of_fg_toSubalgebra


theorem fg_of_noetherian (S : IntermediateField F E) [IsNoetherian F E] : S.FG :=
  S.fg_of_fg_toSubalgebra S.toSubalgebra.fg_of_noetherian


theorem induction_on_adjoin_finset (S : Finset E) (P : IntermediateField F E → Prop) (base : P ⊥)
    (ih : ∀ (K : IntermediateField F E), ∀ x ∈ S, P K → P (K⟮x⟯.restrictScalars F)) :
    P (adjoin F S) := by
  classical
  refine Finset.induction_on' S ?_ (fun ha _ _ h => ?_)
  · simp [base]
  · rw [Finset.coe_insert, Set.insert_eq, Set.union_comm, ← adjoin_adjoin_left]
    exact ih (adjoin F _) _ ha h


theorem induction_on_adjoin_fg (P : IntermediateField F E → Prop) (base : P ⊥)
    (ih : ∀ (K : IntermediateField F E) (x : E), P K → P (K⟮x⟯.restrictScalars F))
    (K : IntermediateField F E) (hK : K.FG) : P K := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    P : IntermediateField F E → Prop
    base : P Bot.bot
    ih : ∀ (K : IntermediateField F E) (x : E), P K → P (IntermediateField.restric …
    K : IntermediateField F E
    hK : K.FG
    ⊢ P K
  -/
  obtain ⟨S, rfl⟩ := hK
  /-
    case intro
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    P : IntermediateField F E → Prop
    base : P Bot.bot
    ih : ∀ (K : IntermediateField F E) (x : E), P K → P (IntermediateField.restric …
    S : Finset E
    ⊢ P (IntermediateField.adjoin F ↑S)
  -/
  exact induction_on_adjoin_finset S P base fun K x _ hK => ih K x hK
  /-
    🎉 no goals
  -/


theorem induction_on_adjoin [FiniteDimensional F E] (P : IntermediateField F E → Prop)
    (base : P ⊥) (ih : ∀ (K : IntermediateField F E) (x : E), P K → P (K⟮x⟯.restrictScalars F))
    (K : IntermediateField F E) : P K :=
  letI : IsNoetherian F E := IsNoetherian.iff_fg.2 inferInstance
  induction_on_adjoin_fg P base ih K K.fg_of_noetherian


/-- The canonical algebraic homomorphism from `AdjoinRoot p` to `AdjoinRoot q`, where
  the polynomial `q : K[X]` divides `p`. -/
noncomputable def algHomOfDvd {p q : K[X]} (hpq : q ∣ p) :
    AdjoinRoot p →ₐ[K] AdjoinRoot q :=
                          /-
                            K : Type ?u.673873
                            L : Type ?u.673876
                            inst✝² : Field K
                            inst✝¹ : Field L
                            inst✝ : Algebra K L
                            p q : Polynomial K
                            hpq : Dvd.dvd q p
                            ⊢ Eq ((Polynomial.aeval (AdjoinRoot.root q)) p) 0
                          -/
  (liftHom p (root q) (by simp only [aeval_eq, mk_eq_zero, hpq]))
                          /-
                            🎉 no goals
                          -/


theorem coe_algHomOfDvd {p q : K[X]} (hpq : q ∣ p) :
                                                     /-
                                                       K : Type ?u.675502
                                                       L : Type ?u.675505
                                                       inst✝² : Field K
                                                       inst✝¹ : Field L
                                                       inst✝ : Algebra K L
                                                       p q : Polynomial K
                                                       hpq : Dvd.dvd q p
                                                       ⊢ Eq ((Polynomial.aeval (AdjoinRoot.root q)) p) 0
                                                     -/
    (algHomOfDvd hpq).toFun = liftHom p (root q) (by simp only [aeval_eq, mk_eq_zero, hpq]) :=
                                                     /-
                                                       🎉 no goals
                                                     -/
  rfl


/-- `algHomOfDvd` sends `AdjoinRoot.root p` to `AdjoinRoot.root q`. -/
theorem algHomOfDvd_apply_root {p q : K[X]} (hpq : q ∣ p) :
    algHomOfDvd hpq (root p) = root q := by
  /-
    K : Type u_1
    inst✝ : Field K
    p q : Polynomial K
    hpq : Dvd.dvd q p
    ⊢ Eq ((AdjoinRoot.algHomOfDvd hpq) (AdjoinRoot.root p)) (AdjoinRoot.root q)
  -/
  rw [algHomOfDvd, liftHom_root]
  /-
    🎉 no goals
  -/


/-- The canonical algebraic equivalence between `AdjoinRoot p` and `AdjoinRoot q`, where
  the two polynomials `p q : K[X]` are equal. -/
noncomputable def algEquivOfEq {p q : K[X]} (hp : p ≠ 0) (h_eq : p = q) :
    AdjoinRoot p ≃ₐ[K] AdjoinRoot q :=
  ofAlgHom (algHomOfDvd (dvd_of_eq h_eq.symm)) (algHomOfDvd (dvd_of_eq h_eq))
    (PowerBasis.algHom_ext (powerBasis (h_eq ▸ hp))
      (by rw [algHomOfDvd, powerBasis_gen (h_eq ▸ hp), AlgHom.coe_comp, Function.comp_apply,
        algHomOfDvd, liftHom_root, liftHom_root, AlgHom.coe_id, id_eq]))
    (PowerBasis.algHom_ext (powerBasis hp)
      (by rw [algHomOfDvd, powerBasis_gen hp, AlgHom.coe_comp, Function.comp_apply, algHomOfDvd,
          liftHom_root, liftHom_root, AlgHom.coe_id, id_eq]))


theorem coe_algEquivOfEq {p q : K[X]} (hp : p ≠ 0) (h_eq : p = q) :
                                                          /-
                                                            K : Type ?u.685249
                                                            L : Type ?u.685252
                                                            inst✝² : Field K
                                                            inst✝¹ : Field L
                                                            inst✝ : Algebra K L
                                                            p q : Polynomial K
                                                            hp : Ne p 0
                                                            h_eq : Eq p q
                                                            ⊢ Eq ((Polynomial.aeval (AdjoinRoot.root q)) p) 0
                                                          -/
    (algEquivOfEq hp h_eq).toFun = liftHom p (root q) (by rw [h_eq, aeval_eq, mk_self]) :=
                                                          /-
                                                            🎉 no goals
                                                          -/
  rfl


theorem algEquivOfEq_toAlgHom {p q : K[X]} (hp : p ≠ 0) (h_eq : p = q) :
                                                             /-
                                                               K : Type ?u.686939
                                                               L : Type ?u.686942
                                                               inst✝² : Field K
                                                               inst✝¹ : Field L
                                                               inst✝ : Algebra K L
                                                               p q : Polynomial K
                                                               hp : Ne p 0
                                                               h_eq : Eq p q
                                                               ⊢ Eq ((Polynomial.aeval (AdjoinRoot.root q)) p) 0
                                                             -/
    (algEquivOfEq hp h_eq).toAlgHom = liftHom p (root q) (by rw [h_eq, aeval_eq, mk_self]) :=
                                                             /-
                                                               🎉 no goals
                                                             -/
  rfl


/-- `algEquivOfEq` sends `AdjoinRoot.root p` to `AdjoinRoot.root q`. -/
theorem algEquivOfEq_apply_root {p q : K[X]} (hp : p ≠ 0) (h_eq : p = q) :
    algEquivOfEq hp h_eq (root p) = root q := by
  /-
    K : Type u_1
    inst✝ : Field K
    p q : Polynomial K
    hp : Ne p 0
    h_eq : Eq p q
    ⊢ Eq ((AdjoinRoot.algEquivOfEq hp h_eq) (AdjoinRoot.root p)) (AdjoinRoot.root q)
  -/
  rw [← coe_algHom, algEquivOfEq_toAlgHom, liftHom_root]
  /-
    🎉 no goals
  -/


/-- The canonical algebraic equivalence between `AdjoinRoot p` and `AdjoinRoot q`,
where the two polynomials `p q : K[X]` are associated.-/
noncomputable def algEquivOfAssociated {p q : K[X]} (hp : p ≠ 0) (hpq : Associated p q) :
    AdjoinRoot p ≃ₐ[K] AdjoinRoot q :=
                                   /-
                                     K : Type ?u.690508
                                     L : Type ?u.690511
                                     inst✝² : Field K
                                     inst✝¹ : Field L
                                     inst✝ : Algebra K L
                                     p q : Polynomial K
                                     hp : Ne p 0
                                     hpq : Associated p q
                                     ⊢ Eq ((Polynomial.aeval (AdjoinRoot.root q)) p) 0
                                   -/
  ofAlgHom (liftHom p (root q) (by simp only [aeval_eq, mk_eq_zero, hpq.symm.dvd] ))
                                   /-
                                     🎉 no goals
                                   -/
                            /-
                              K : Type ?u.690508
                              L : Type ?u.690511
                              inst✝² : Field K
                              inst✝¹ : Field L
                              inst✝ : Algebra K L
                              p q : Polynomial K
                              hp : Ne p 0
                              hpq : Associated p q
                              ⊢ Eq ((Polynomial.aeval (AdjoinRoot.root p)) q) 0
                            -/
    (liftHom q (root p) (by simp only [aeval_eq, mk_eq_zero, hpq.dvd]))
                            /-
                              🎉 no goals
                            -/
    ( PowerBasis.algHom_ext (powerBasis (hpq.ne_zero_iff.mp hp))
        (by rw [powerBasis_gen (hpq.ne_zero_iff.mp hp), AlgHom.coe_comp, Function.comp_apply,
          liftHom_root, liftHom_root, AlgHom.coe_id, id_eq]))
    (PowerBasis.algHom_ext (powerBasis hp)
      (by rw [powerBasis_gen hp, AlgHom.coe_comp, Function.comp_apply, liftHom_root, liftHom_root,
          AlgHom.coe_id, id_eq]))


theorem coe_algEquivOfAssociated {p q : K[X]} (hp : p ≠ 0) (hpq : Associated p q) :
    (algEquivOfAssociated hp hpq).toFun =
                             /-
                               K : Type ?u.696713
                               L : Type ?u.696716
                               inst✝² : Field K
                               inst✝¹ : Field L
                               inst✝ : Algebra K L
                               p q : Polynomial K
                               hp : Ne p 0
                               hpq : Associated p q
                               ⊢ Eq ((Polynomial.aeval (AdjoinRoot.root q)) p) 0
                             -/
      liftHom p (root q) (by simp only [aeval_eq, mk_eq_zero, hpq.symm.dvd]) :=
                             /-
                               🎉 no goals
                             -/
  rfl


theorem algEquivOfAssociated_toAlgHom {p q : K[X]} (hp : p ≠ 0) (hpq : Associated p q) :
    (algEquivOfAssociated hp hpq).toAlgHom =
                             /-
                               K : Type ?u.698524
                               L : Type ?u.698527
                               inst✝² : Field K
                               inst✝¹ : Field L
                               inst✝ : Algebra K L
                               p q : Polynomial K
                               hp : Ne p 0
                               hpq : Associated p q
                               ⊢ Eq ((Polynomial.aeval (AdjoinRoot.root q)) p) 0
                             -/
      liftHom p (root q) (by simp only [aeval_eq, mk_eq_zero, hpq.symm.dvd]) :=
                             /-
                               🎉 no goals
                             -/
  rfl


/-- `algEquivOfAssociated` sends `AdjoinRoot.root p` to `AdjoinRoot.root q`. -/
theorem algEquivOfAssociated_apply_root {p q : K[X]} (hp : p ≠ 0) (hpq : Associated p q) :
    algEquivOfAssociated hp hpq (root p) = root q := by
  /-
    K : Type u_1
    inst✝ : Field K
    p q : Polynomial K
    hp : Ne p 0
    hpq : Associated p q
    ⊢ Eq ((AdjoinRoot.algEquivOfAssociated hp hpq) (AdjoinRoot.root p)) (AdjoinRoo …
  -/
  rw [← coe_algHom, algEquivOfAssociated_toAlgHom, liftHom_root]
  /-
    🎉 no goals
  -/


/-- If `y : L` is a root of `minpoly K x`, then `minpoly K y = minpoly K x`. -/
theorem eq_of_root {x y : L} (hx : IsAlgebraic K x)
    (h_ev : Polynomial.aeval y (minpoly K x) = 0) : minpoly K y = minpoly K x :=
  ((eq_iff_aeval_minpoly_eq_zero hx.isIntegral).mpr h_ev).symm


/-- The canonical `algEquiv` between `K⟮x⟯`and `K⟮y⟯`, sending `x` to `y`, where `x` and `y` have
  the same minimal polynomial over `K`. -/
noncomputable def algEquiv {x y : L} (hx : IsAlgebraic K x)
    (h_mp : minpoly K x = minpoly K y) : K⟮x⟯ ≃ₐ[K] K⟮y⟯ := by
  /-
    K : Type ?u.704097
    L : Type ?u.704100
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x y : L
    hx : IsAlgebraic K x
    h_mp : Eq (minpoly K x) (minpoly K y)
    ⊢ AlgEquiv K (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin K (S …
  -/
  have hy : IsAlgebraic K y := ⟨minpoly K x, ne_zero hx.isIntegral, (h_mp ▸ aeval _ _)⟩
  exact AlgEquiv.trans (adjoinRootEquivAdjoin K hx.isIntegral).symm
    (AlgEquiv.trans (AdjoinRoot.algEquivOfEq (ne_zero hx.isIntegral) h_mp)
      (adjoinRootEquivAdjoin K hy.isIntegral))


/-- `minpoly.algEquiv` sends the generator of `K⟮x⟯` to the generator of `K⟮y⟯`. -/
theorem algEquiv_apply {x y : L} (hx : IsAlgebraic K x) (h_mp : minpoly K x = minpoly K y) :
    algEquiv hx h_mp (AdjoinSimple.gen K x) = AdjoinSimple.gen K y := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x y : L
    hx : IsAlgebraic K x
    h_mp : Eq (minpoly K x) (minpoly K y)
    ⊢ Eq ((minpoly.algEquiv hx h_mp) (IntermediateField.AdjoinSimple.gen K x)) (In …
  -/
  have hy : IsAlgebraic K y := ⟨minpoly K x, ne_zero hx.isIntegral, (h_mp ▸ aeval _ _)⟩
  rw [algEquiv, trans_apply, ← adjoinRootEquivAdjoin_apply_root K hx.isIntegral,
    symm_apply_apply, trans_apply, AdjoinRoot.algEquivOfEq_apply_root,
    adjoinRootEquivAdjoin_apply_root K hy.isIntegral]


/-- `pb.equivAdjoinSimple` is the equivalence between `K⟮pb.gen⟯` and `L` itself. -/
noncomputable def equivAdjoinSimple (pb : PowerBasis K L) : K⟮pb.gen⟯ ≃ₐ[K] L :=
  (adjoin.powerBasis pb.isIntegral_gen).equivOfMinpoly pb <| by
    /-
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      pb : PowerBasis K L
      ⊢ Eq (minpoly K (IntermediateField.adjoin.powerBasis ⋯).gen) (minpoly K pb.gen)
    -/
    rw [adjoin.powerBasis_gen, minpoly_gen]
    /-
      🎉 no goals
    -/


@[simp]
theorem equivAdjoinSimple_aeval (pb : PowerBasis K L) (f : K[X]) :
    pb.equivAdjoinSimple (aeval (AdjoinSimple.gen K pb.gen) f) = aeval pb.gen f :=
  equivOfMinpoly_aeval _ pb _ f


@[simp]
theorem equivAdjoinSimple_gen (pb : PowerBasis K L) :
    pb.equivAdjoinSimple (AdjoinSimple.gen K pb.gen) = pb.gen :=
  equivOfMinpoly_gen _ pb _


@[simp]
theorem equivAdjoinSimple_symm_aeval (pb : PowerBasis K L) (f : K[X]) :
    pb.equivAdjoinSimple.symm (aeval pb.gen f) = aeval (AdjoinSimple.gen K pb.gen) f := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    pb : PowerBasis K L
    f : Polynomial K
    ⊢ Eq (pb.equivAdjoinSimple.symm ((Polynomial.aeval pb.gen) f)) ((Polynomial.ae …
  -/
  rw [equivAdjoinSimple, equivOfMinpoly_symm, equivOfMinpoly_aeval, adjoin.powerBasis_gen]
  /-
    🎉 no goals
  -/


@[simp]
theorem equivAdjoinSimple_symm_gen (pb : PowerBasis K L) :
    pb.equivAdjoinSimple.symm pb.gen = AdjoinSimple.gen K pb.gen := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    pb : PowerBasis K L
    ⊢ Eq (pb.equivAdjoinSimple.symm pb.gen) (IntermediateField.AdjoinSimple.gen K  …
  -/
  rw [equivAdjoinSimple, equivOfMinpoly_symm, equivOfMinpoly_gen, adjoin.powerBasis_gen]
  /-
    🎉 no goals
  -/


theorem map_comap_eq (f : L →ₐ[K] L') (S : IntermediateField K L') :
    (S.comap f).map f = S ⊓ f.fieldRange :=
  SetLike.coe_injective Set.image_preimage_eq_inter_range


theorem map_comap_eq_self {f : L →ₐ[K] L'} {S : IntermediateField K L'} (h : S ≤ f.fieldRange) :
    (S.comap f).map f = S := by
  /-
    K : Type u_1
    L : Type u_2
    L' : Type u_3
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Field L'
    inst✝¹ : Algebra K L
    inst✝ : Algebra K L'
    f : AlgHom K L L'
    S : IntermediateField K L'
    h : LE.le S f.fieldRange
    ⊢ Eq (IntermediateField.map f (IntermediateField.comap f S)) S
  -/
  simpa only [inf_of_le_left h] using map_comap_eq f S
  /-
    🎉 no goals
  -/


theorem map_comap_eq_self_of_surjective {f : L →ₐ[K] L'} (hf : Function.Surjective f)
    (S : IntermediateField K L') : (S.comap f).map f = S :=
  SetLike.coe_injective (Set.image_preimage_eq _ hf)


theorem comap_map (f : L →ₐ[K] L') (S : IntermediateField K L) : (S.map f).comap f = S :=
  SetLike.coe_injective (Set.preimage_image_eq _ f.injective)


@[simp]
theorem extendScalars_self : extendScalars (le_refl F) = ⊥ := by
  /-
    L : Type u_2
    inst✝ : Field L
    F : Subfield L
    ⊢ Eq (Subfield.extendScalars ⋯) Bot.bot
  -/
  ext x
  /-
    case h
    L : Type u_2
    inst✝ : Field L
    F : Subfield L
    x : L
    ⊢ Iff (Membership.mem (Subfield.extendScalars ⋯) x) (Membership.mem Bot.bot x)
  -/
  rw [mem_extendScalars, IntermediateField.mem_bot]
  /-
    case h
    L : Type u_2
    inst✝ : Field L
    F : Subfield L
    x : L
    ⊢ Iff (Membership.mem F x) (Membership.mem (Set.range ⇑(algebraMap (Subtype fu …
  -/
  refine ⟨fun h ↦ ⟨⟨x, h⟩, rfl⟩, ?_⟩
  /-
    case h
    L : Type u_2
    inst✝ : Field L
    F : Subfield L
    x : L
    ⊢ Membership.mem (Set.range ⇑(algebraMap (Subtype fun x => Membership.mem F x) …
  -/
  rintro ⟨y, rfl⟩
  /-
    case h.intro
    L : Type u_2
    inst✝ : Field L
    F : Subfield L
    y : Subtype fun x => Membership.mem F x
    ⊢ Membership.mem F ((algebraMap (Subtype fun x => Membership.mem F x) L) y)
  -/
  exact y.2
  /-
    🎉 no goals
  -/


@[simp]
theorem extendScalars_top : extendScalars (le_top : F ≤ ⊤) = ⊤ :=
                                             /-
                                               L : Type u_2
                                               inst✝ : Field L
                                               F : Subfield L
                                               ⊢ Eq (Subfield.extendScalars ⋯).toSubfield Top.top.toSubfield
                                             -/
  IntermediateField.toSubfield_injective (by simp)
                                             /-
                                               🎉 no goals
                                             -/


theorem extendScalars_sup :
    extendScalars h ⊔ extendScalars h' = extendScalars (le_sup_of_le_left h : F ≤ E ⊔ E') :=
  ((extendScalars.orderIso F).map_sup ⟨_, h⟩ ⟨_, h'⟩).symm


theorem extendScalars_inf : extendScalars h ⊓ extendScalars h' = extendScalars (le_inf h h') :=
  ((extendScalars.orderIso F).map_inf ⟨_, h⟩ ⟨_, h'⟩).symm


@[simp]
theorem extendScalars_self : extendScalars (le_refl F) = ⊥ :=
                                  /-
                                    K : Type u_1
                                    inst✝² : Field K
                                    L : Type u_2
                                    inst✝¹ : Field L
                                    inst✝ : Algebra K L
                                    F : IntermediateField K L
                                    ⊢ Eq (IntermediateField.restrictScalars K (IntermediateField.extendScalars ⋯)) …
                                  -/
  restrictScalars_injective K (by simp)
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem extendScalars_top : extendScalars (le_top : F ≤ ⊤) = ⊤ :=
                                  /-
                                    K : Type u_1
                                    inst✝² : Field K
                                    L : Type u_2
                                    inst✝¹ : Field L
                                    inst✝ : Algebra K L
                                    F : IntermediateField K L
                                    ⊢ Eq (IntermediateField.restrictScalars K (IntermediateField.extendScalars ⋯)) …
                                  -/
  restrictScalars_injective K (by simp)
                                  /-
                                    🎉 no goals
                                  -/


/-- If `F` is a field, `A` is an `F`-algebra with fraction field `K`, `L` is a field,
`g : A →ₐ[F] L` lifts to `f : K →ₐ[F] L`,
then the image of `f` is the field generated by the image of `g`.
Note: this does not require `IsScalarTower F A K`. -/
theorem algHom_fieldRange_eq_of_comp_eq (h : RingHom.comp f (algebraMap A K) = (g : A →+* L)) :
    f.fieldRange = IntermediateField.adjoin F g.range := by
  /-
    F : Type u_1
    A : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝⁸ : Field F
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra F A
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : Field L
    inst✝ : Algebra F L
    g : AlgHom F A L
    f : AlgHom F K L
    h : Eq ((↑f).comp (algebraMap A K)) ↑g
    ⊢ Eq f.fieldRange (IntermediateField.adjoin F ↑g.range)
  -/
  apply IntermediateField.toSubfield_injective
  /-
    case a
    F : Type u_1
    A : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝⁸ : Field F
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra F A
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : Field L
    inst✝ : Algebra F L
    g : AlgHom F A L
    f : AlgHom F K L
    h : Eq ((↑f).comp (algebraMap A K)) ↑g
    ⊢ Eq f.fieldRange.toSubfield (IntermediateField.adjoin F ↑g.range).toSubfield
  -/
  simp_rw [AlgHom.fieldRange_toSubfield, IntermediateField.adjoin_toSubfield]
  /-
    case a
    F : Type u_1
    A : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝⁸ : Field F
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra F A
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : Field L
    inst✝ : Algebra F L
    g : AlgHom F A L
    f : AlgHom F K L
    h : Eq ((↑f).comp (algebraMap A K)) ↑g
    ⊢ Eq (↑f).fieldRange (Subfield.closure (Union.union (Set.range ⇑(algebraMap F  …
  -/
  convert ringHom_fieldRange_eq_of_comp_eq h using 2
  /-
    case h.e'_3.h.e'_3
    F : Type u_1
    A : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝⁸ : Field F
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra F A
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : Field L
    inst✝ : Algebra F L
    g : AlgHom F A L
    f : AlgHom F K L
    h : Eq ((↑f).comp (algebraMap A K)) ↑g
    ⊢ Eq (Union.union (Set.range ⇑(algebraMap F L)) ↑g.range) ↑(↑g).range
  -/
  exact Set.union_eq_self_of_subset_left fun _ ⟨x, hx⟩ ↦ ⟨algebraMap F A x, by simp [← hx]⟩
  /-
    🎉 no goals
  -/


/-- If `F` is a field, `A` is an `F`-algebra with fraction field `K`, `L` is a field,
`g : A →ₐ[F] L` lifts to `f : K →ₐ[F] L`,
`s` is a set such that the image of `g` is the subalgebra generated by `s`,
then the image of `f` is the intermediate field generated by `s`.
Note: this does not require `IsScalarTower F A K`. -/
theorem algHom_fieldRange_eq_of_comp_eq_of_range_eq
    (h : RingHom.comp f (algebraMap A K) = (g : A →+* L))
    {s : Set L} (hs : g.range = Algebra.adjoin F s) :
    f.fieldRange = IntermediateField.adjoin F s := by
  /-
    F : Type u_1
    A : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝⁸ : Field F
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra F A
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : Field L
    inst✝ : Algebra F L
    g : AlgHom F A L
    f : AlgHom F K L
    h : Eq ((↑f).comp (algebraMap A K)) ↑g
    s : Set L
    hs : Eq g.range (Algebra.adjoin F s)
    ⊢ Eq f.fieldRange (IntermediateField.adjoin F s)
  -/
  apply IntermediateField.toSubfield_injective
  /-
    case a
    F : Type u_1
    A : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝⁸ : Field F
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra F A
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : Field L
    inst✝ : Algebra F L
    g : AlgHom F A L
    f : AlgHom F K L
    h : Eq ((↑f).comp (algebraMap A K)) ↑g
    s : Set L
    hs : Eq g.range (Algebra.adjoin F s)
    ⊢ Eq f.fieldRange.toSubfield (IntermediateField.adjoin F s).toSubfield
  -/
  simp_rw [AlgHom.fieldRange_toSubfield, IntermediateField.adjoin_toSubfield]
  /-
    case a
    F : Type u_1
    A : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝⁸ : Field F
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra F A
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : Field L
    inst✝ : Algebra F L
    g : AlgHom F A L
    f : AlgHom F K L
    h : Eq ((↑f).comp (algebraMap A K)) ↑g
    s : Set L
    hs : Eq g.range (Algebra.adjoin F s)
    ⊢ Eq (↑f).fieldRange (Subfield.closure (Union.union (Set.range ⇑(algebraMap F  …
  -/
  refine ringHom_fieldRange_eq_of_comp_eq_of_range_eq h ?_
  /-
    case a
    F : Type u_1
    A : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝⁸ : Field F
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra F A
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : Field L
    inst✝ : Algebra F L
    g : AlgHom F A L
    f : AlgHom F K L
    h : Eq ((↑f).comp (algebraMap A K)) ↑g
    s : Set L
    hs : Eq g.range (Algebra.adjoin F s)
    ⊢ Eq (↑g).range (Subring.closure (Union.union (Set.range ⇑(algebraMap F L)) s))
  -/
  rw [← Algebra.adjoin_eq_ring_closure, ← hs]; rfl
                                               /-
                                                 🎉 no goals
                                               -/


/-- The image of `IsFractionRing.liftAlgHom` is the intermediate field generated by the image
of the algebra hom. -/
theorem liftAlgHom_fieldRange (hg : Function.Injective g) :
    (liftAlgHom hg : K →ₐ[F] L).fieldRange = IntermediateField.adjoin F g.range :=
                                      /-
                                        F : Type u_1
                                        A : Type u_2
                                        K : Type u_3
                                        L : Type u_4
                                        inst✝⁹ : Field F
                                        inst✝⁸ : CommRing A
                                        inst✝⁷ : Algebra F A
                                        inst✝⁶ : Field K
                                        inst✝⁵ : Algebra F K
                                        inst✝⁴ : Algebra A K
                                        inst✝³ : IsFractionRing A K
                                        inst✝² : Field L
                                        inst✝¹ : Algebra F L
                                        g : AlgHom F A L
                                        inst✝ : IsScalarTower F A K
                                        hg : Function.Injective ⇑g
                                        ⊢ Eq ((↑(IsFractionRing.liftAlgHom hg)).comp (algebraMap A K)) ↑g
                                      -/
  algHom_fieldRange_eq_of_comp_eq (by ext; simp)
                                           /-
                                             🎉 no goals
                                           -/


/-- The image of `IsFractionRing.liftAlgHom` is the intermediate field generated by `s`,
if the image of the algebra hom is the subalgebra generated by `s`. -/
theorem liftAlgHom_fieldRange_eq_of_range_eq (hg : Function.Injective g)
    {s : Set L} (hs : g.range = Algebra.adjoin F s) :
    (liftAlgHom hg : K →ₐ[F] L).fieldRange = IntermediateField.adjoin F s :=
                                                  /-
                                                    F : Type u_1
                                                    A : Type u_2
                                                    K : Type u_3
                                                    L : Type u_4
                                                    inst✝⁹ : Field F
                                                    inst✝⁸ : CommRing A
                                                    inst✝⁷ : Algebra F A
                                                    inst✝⁶ : Field K
                                                    inst✝⁵ : Algebra F K
                                                    inst✝⁴ : Algebra A K
                                                    inst✝³ : IsFractionRing A K
                                                    inst✝² : Field L
                                                    inst✝¹ : Algebra F L
                                                    g : AlgHom F A L
                                                    inst✝ : IsScalarTower F A K
                                                    hg : Function.Injective ⇑g
                                                    s : Set L
                                                    hs : Eq g.range (Algebra.adjoin F s)
                                                    ⊢ Eq ((↑(IsFractionRing.liftAlgHom hg)).comp (algebraMap A K)) ↑g
                                                  -/
  algHom_fieldRange_eq_of_comp_eq_of_range_eq (by ext; simp) hs
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem lift_cardinalMk_adjoin_le {E : Type v} [Field E] [Algebra F E] (s : Set E) :
    Cardinal.lift.{u} #(adjoin F s) ≤ Cardinal.lift.{v} #F ⊔ Cardinal.lift.{u} #s ⊔ ℵ₀ := by
  /-
    F : Type u
    inst✝² : Field F
    E : Type v
    inst✝¹ : Field E
    inst✝ : Algebra F E
    s : Set E
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk (Subtype fun x => Membership.mem (I …
  -/
  rw [show ↥(adjoin F s) = (adjoin F s).toSubfield from rfl, adjoin_toSubfield]
  /-
    F : Type u
    inst✝² : Field F
    E : Type v
    inst✝¹ : Field E
    inst✝ : Algebra F E
    s : Set E
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk (Subtype fun x => Membership.mem (S …
  -/
  apply (Cardinal.lift_le.mpr (Subfield.cardinalMk_closure_le_max _)).trans
  /-
    F : Type u
    inst✝² : Field F
    E : Type v
    inst✝¹ : Field E
    inst✝ : Algebra F E
    s : Set E
    ⊢ LE.le (Cardinal.lift.{u, v} (Max.max (Cardinal.mk ↑(Union.union (Set.range ⇑ …
  -/
  rw [lift_max, sup_le_iff, lift_aleph0]
  /-
    F : Type u
    inst✝² : Field F
    E : Type v
    inst✝¹ : Field E
    inst✝ : Algebra F E
    s : Set E
    ⊢ And (LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑(Union.union (Set.range ⇑(alg …
  -/
  refine ⟨(Cardinal.lift_le.mpr ((mk_union_le _ _).trans <| add_le_max _ _)).trans ?_, le_sup_right⟩
  /-
    F : Type u
    inst✝² : Field F
    E : Type v
    inst✝¹ : Field E
    inst✝ : Algebra F E
    s : Set E
    ⊢ LE.le (Cardinal.lift.{u, v} (Max.max (Max.max (Cardinal.mk ↑(Set.range ⇑(alg …
  -/
  simp_rw [lift_max, lift_aleph0, sup_assoc]
  /-
    F : Type u
    inst✝² : Field F
    E : Type v
    inst✝¹ : Field E
    inst✝ : Algebra F E
    s : Set E
    ⊢ LE.le (Max.max (Cardinal.lift.{u, v} (Cardinal.mk ↑(Set.range ⇑(algebraMap F …
  -/
  exact sup_le_sup_right mk_range_le_lift _
  /-
    🎉 no goals
  -/


theorem cardinalMk_adjoin_le {E : Type u} [Field E] [Algebra F E] (s : Set E) :
    #(adjoin F s) ≤ #F ⊔ #s ⊔ ℵ₀ := by
  /-
    F : Type u
    inst✝² : Field F
    E : Type u
    inst✝¹ : Field E
    inst✝ : Algebra F E
    s : Set E
    ⊢ LE.le (Cardinal.mk (Subtype fun x => Membership.mem (IntermediateField.adjoi …
  -/
  simpa using lift_cardinalMk_adjoin_le F s
  /-
    🎉 no goals
  -/


