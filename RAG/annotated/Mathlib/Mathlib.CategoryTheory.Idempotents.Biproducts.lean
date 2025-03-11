/-- The `Bicone` used in order to obtain the existence of
the biproduct of a functor `J ⥤ Karoubi C` when the category `C` is additive. -/
@[simps]
def bicone [HasFiniteBiproducts C] {J : Type} [Finite J] (F : J → Karoubi C) : Bicone F where
  pt :=
    { X := biproduct fun j => (F j).X
      p := biproduct.map fun j => (F j).p
      idem := by
        /-
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u_1} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
          J : Type
          inst✝ : Finite J
          F : J → CategoryTheory.Idempotents.Karoubi C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.map  …
        -/
        ext
        /-
          case w.w
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u_1} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
          J : Type
          inst✝ : Finite J
          F : J → CategoryTheory.Idempotents.Karoubi C
          j✝¹ j✝ : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
        -/
        simp only [assoc, biproduct.map_π, biproduct.map_π_assoc, idem] }
        /-
          🎉 no goals
        -/
  π j :=
    { f := (biproduct.map fun j => (F j).p) ≫ Bicone.π _ j
      comm := by
        /-
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u_1} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
          J : Type
          inst✝ : Finite J
          F : J → CategoryTheory.Idempotents.Karoubi C
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.map  …
        -/
        simp only [assoc, biproduct.bicone_π, biproduct.map_π, biproduct.map_π_assoc, (F j).idem] }
        /-
          🎉 no goals
        -/
  ι j :=
    { f := biproduct.ι (fun j => (F j).X) j ≫ biproduct.map fun j => (F j).p
                 /-
                   C : Type u_1
                   inst✝³ : CategoryTheory.Category.{v, u_1} C
                   inst✝² : CategoryTheory.Preadditive C
                   inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
                   J : Type
                   inst✝ : Finite J
                   F : J → CategoryTheory.Idempotents.Karoubi C
                   j : J
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
                 -/
      comm := by simp only [biproduct.ι_map, assoc, idem_assoc] }
                 /-
                   🎉 no goals
                 -/
  ι_π j j' := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{v, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
      J : Type
      inst✝ : Finite J
      F : J → CategoryTheory.Idempotents.Karoubi C
      j j' : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => { f := CategoryTheory.Cate …
    -/
    split_ifs with h
      /-
        case pos
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{v, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
        J : Type
        inst✝ : Finite J
        F : J → CategoryTheory.Idempotents.Karoubi C
        j j' : J
        h : Eq j j'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => { f := CategoryTheory.Cate …
      -/
    · subst h
      simp only [biproduct.ι_map, biproduct.bicone_π, biproduct.map_π, eqToHom_refl,
        id_f, hom_ext_iff, comp_f, assoc, bicone_ι_π_self_assoc, idem]
      /-
        case neg
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{v, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
        J : Type
        inst✝ : Finite J
        F : J → CategoryTheory.Idempotents.Karoubi C
        j j' : J
        h : Not (Eq j j')
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => { f := CategoryTheory.Cate …
      -/
    · dsimp
      simp only [biproduct.ι_map, biproduct.map_π, hom_ext_iff, comp_f,
        assoc, biproduct.ι_π_ne_assoc _ h, zero_comp, comp_zero, instZero_zero]


theorem karoubi_hasFiniteBiproducts [HasFiniteBiproducts C] : HasFiniteBiproducts (Karoubi C) :=
  { out := fun n =>
      { has_biproduct := fun F => by
          /-
            C : Type u_1
            inst✝² : CategoryTheory.Category.{v, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteBiproducts C
            n : Nat
            F : Fin n → CategoryTheory.Idempotents.Karoubi C
            ⊢ CategoryTheory.Limits.HasBiproduct F
          -/
          apply hasBiproduct_of_total (Biproducts.bicone F)
          /-
            C : Type u_1
            inst✝² : CategoryTheory.Category.{v, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteBiproducts C
            n : Nat
            F : Fin n → CategoryTheory.Idempotents.Karoubi C
            ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp ((CategoryTh …
          -/
          simp only [hom_ext_iff]
          /-
            C : Type u_1
            inst✝² : CategoryTheory.Category.{v, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteBiproducts C
            n : Nat
            F : Fin n → CategoryTheory.Idempotents.Karoubi C
            ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp ((CategoryTh …
          -/
          refine biproduct.hom_ext' _ _ (fun j => ?_)
          simp only [Biproducts.bicone_pt_X, sum_hom, comp_f, Biproducts.bicone_π_f,
            biproduct.bicone_π, biproduct.map_π, Biproducts.bicone_ι_f, biproduct.ι_map, assoc,
            idem_assoc, id_f, Biproducts.bicone_pt_p, comp_sum]
          /-
            C : Type u_1
            inst✝² : CategoryTheory.Category.{v, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteBiproducts C
            n : Nat
            F : Fin n → CategoryTheory.Idempotents.Karoubi C
            j : Fin n
            ⊢ Eq (Finset.univ.sum fun j_1 => CategoryTheory.CategoryStruct.comp (CategoryT …
          -/
          rw [Finset.sum_eq_single j]
            /-
              C : Type u_1
              inst✝² : CategoryTheory.Category.{v, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteBiproducts C
              n : Nat
              F : Fin n → CategoryTheory.Idempotents.Karoubi C
              j : Fin n
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
            -/
          · simp only [bicone_ι_π_self_assoc]
            /-
              🎉 no goals
            -/
            /-
              case h₀
              C : Type u_1
              inst✝² : CategoryTheory.Category.{v, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteBiproducts C
              n : Nat
              F : Fin n → CategoryTheory.Idempotents.Karoubi C
              j : Fin n
              ⊢ ∀ (b : Fin n), Membership.mem Finset.univ b → Ne b j → Eq (CategoryTheory.Ca …
            -/
          · intro b _ hb
            /-
              case h₀
              C : Type u_1
              inst✝² : CategoryTheory.Category.{v, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteBiproducts C
              n : Nat
              F : Fin n → CategoryTheory.Idempotents.Karoubi C
              j b : Fin n
              a✝ : Membership.mem Finset.univ b
              hb : Ne b j
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
            -/
            simp only [biproduct.ι_π_ne_assoc _ hb.symm, zero_comp]
            /-
              🎉 no goals
            -/
            /-
              case h₁
              C : Type u_1
              inst✝² : CategoryTheory.Category.{v, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteBiproducts C
              n : Nat
              F : Fin n → CategoryTheory.Idempotents.Karoubi C
              j : Fin n
              ⊢ Not (Membership.mem Finset.univ j) → Eq (CategoryTheory.CategoryStruct.comp  …
            -/
          · intro hj
            /-
              case h₁
              C : Type u_1
              inst✝² : CategoryTheory.Category.{v, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteBiproducts C
              n : Nat
              F : Fin n → CategoryTheory.Idempotents.Karoubi C
              j : Fin n
              hj : Not (Membership.mem Finset.univ j)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
            -/
            simp only [Finset.mem_univ, not_true] at hj } }
            /-
              🎉 no goals
            -/


/-- `P.complement` is the formal direct factor of `P.X` given by the idempotent
endomorphism `𝟙 P.X - P.p` -/
@[simps]
def complement (P : Karoubi C) : Karoubi C where
  X := P.X
  p := 𝟙 _ - P.p
  idem := idem_of_id_sub_idem P.p P.idem


instance (P : Karoubi C) : HasBinaryBiproduct P P.complement :=
  hasBinaryBiproduct_of_total
    { pt := P.X
      fst := P.decompId_p
      snd := P.complement.decompId_p
      inl := P.decompId_i
      inr := P.complement.decompId_i
      inl_fst := P.decompId.symm
      inl_snd := by
        simp only [instZero_zero, hom_ext_iff, complement_X, comp_f,
          decompId_i_f, decompId_p_f, complement_p, comp_sub, comp_id, idem, sub_self]
      inr_fst := by
        simp only [instZero_zero, hom_ext_iff, complement_X, comp_f,
          decompId_i_f, complement_p, decompId_p_f, sub_comp, id_comp, idem, sub_self]
      inr_snd := P.complement.decompId.symm }
    (by
      /-
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{v, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        P : CategoryTheory.Idempotents.Karoubi C
        ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp { pt := { X := P.X, p := C …
      -/
      ext
      simp only [complement_X, comp_f, decompId_i_f, decompId_p_f, complement_p, instAdd_add, idem,
        comp_sub, comp_id, sub_comp, id_comp, sub_self, sub_zero, add_sub_cancel, id_f])


/-- A formal direct factor `P : Karoubi C` of an object `P.X : C` in a
preadditive category is actually a direct factor of the image `(toKaroubi C).obj P.X`
of `P.X` in the category `Karoubi C` -/
def decomposition (P : Karoubi C) : P ⊞ P.complement ≅ (toKaroubi _).obj P.X where
  hom := biprod.desc P.decompId_i P.complement.decompId_i
  inv := biprod.lift P.decompId_p P.complement.decompId_p
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{v, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P : CategoryTheory.Idempotents.Karoubi C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.desc P. …
    -/
    apply biprod.hom_ext'
    · rw [biprod.inl_desc_assoc, comp_id, biprod.lift_eq, comp_add, ← decompId_assoc,
        add_right_eq_self, ← assoc]
      /-
        case h₀
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{v, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        P : CategoryTheory.Idempotents.Karoubi C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp P …
      -/
      refine (?_ =≫ _).trans zero_comp
      /-
        case h₀
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{v, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        P : CategoryTheory.Idempotents.Karoubi C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp P.decompId_i P.complement.decompId_p) 0
      -/
      ext
      simp only [comp_f, toKaroubi_obj_X, decompId_i_f, decompId_p_f,
        complement_p, comp_sub, comp_id, idem, sub_self, instZero_zero]
    · rw [biprod.inr_desc_assoc, comp_id, biprod.lift_eq, comp_add, ← decompId_assoc,
        add_left_eq_self, ← assoc]
      /-
        case h₁
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{v, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        P : CategoryTheory.Idempotents.Karoubi C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp P …
      -/
      refine (?_ =≫ _).trans zero_comp
      /-
        case h₁
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{v, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        P : CategoryTheory.Idempotents.Karoubi C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp P.complement.decompId_i P.decompId_p) 0
      -/
      ext
      simp only [complement_X, comp_f, decompId_i_f, complement_p,
        decompId_p_f, sub_comp, id_comp, idem, sub_self, instZero_zero]
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{v, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P : CategoryTheory.Idempotents.Karoubi C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift P. …
    -/
    ext
    simp only [toKaroubi_obj_X, biprod.lift_desc, instAdd_add, comp_f, decompId_p_f, decompId_i_f,
      idem, complement_X, complement_p, comp_sub, comp_id, sub_comp, id_comp, sub_self, sub_zero,
      add_sub_cancel, id_f, toKaroubi_obj_p]


