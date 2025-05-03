/-- Given a nilpotent Lie subalgebra `H ⊆ L`, the root space of a map `χ : H → R` is the weight
space of `L` regarded as a module of `H` via the adjoint action. -/
abbrev rootSpace (χ : H → R) : LieSubmodule R H L :=
  genWeightSpace L χ


theorem zero_rootSpace_eq_top_of_nilpotent [IsNilpotent R L] :
    rootSpace (⊤ : LieSubalgebra R L) 0 = ⊤ :=
  zero_genWeightSpace_eq_top_of_nilpotent L


@[simp]
theorem rootSpace_comap_eq_genWeightSpace (χ : H → R) :
    (rootSpace H χ).comap H.incl' = genWeightSpace H χ :=
  comap_genWeightSpace_eq_of_injective Subtype.coe_injective


theorem lie_mem_genWeightSpace_of_mem_genWeightSpace {χ₁ χ₂ : H → R} {x : L} {m : M}
    (hx : x ∈ rootSpace H χ₁) (hm : m ∈ genWeightSpace M χ₂) :
    ⁅x, m⁆ ∈ genWeightSpace M (χ₁ + χ₂) := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    M : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    χ₁ χ₂ : (Subtype fun x => Membership.mem H x) → R
    x : L
    m : M
    hx : Membership.mem (LieAlgebra.rootSpace H χ₁) x
    hm : Membership.mem (LieModule.genWeightSpace M χ₂) m
    ⊢ Membership.mem (LieModule.genWeightSpace M (HAdd.hAdd χ₁ χ₂)) (Bracket.brack …
  -/
  rw [genWeightSpace, LieSubmodule.mem_iInf]
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    M : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    χ₁ χ₂ : (Subtype fun x => Membership.mem H x) → R
    x : L
    m : M
    hx : Membership.mem (LieAlgebra.rootSpace H χ₁) x
    hm : Membership.mem (LieModule.genWeightSpace M χ₂) m
    ⊢ ∀ (i : Subtype fun x => Membership.mem H x), Membership.mem (LieModule.genWe …
  -/
  intro y
  replace hx : x ∈ genWeightSpaceOf L (χ₁ y) y := by
    rw [rootSpace, genWeightSpace, LieSubmodule.mem_iInf] at hx; exact hx y
  replace hm : m ∈ genWeightSpaceOf M (χ₂ y) y := by
    rw [genWeightSpace, LieSubmodule.mem_iInf] at hm; exact hm y
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    M : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    χ₁ χ₂ : (Subtype fun x => Membership.mem H x) → R
    x : L
    m : M
    y : Subtype fun x => Membership.mem H x
    hx : Membership.mem (LieModule.genWeightSpaceOf L (χ₁ y) y) x
    hm : Membership.mem (LieModule.genWeightSpaceOf M (χ₂ y) y) m
    ⊢ Membership.mem (LieModule.genWeightSpaceOf M (HAdd.hAdd χ₁ χ₂ y) y) (Bracket …
  -/
  exact lie_mem_maxGenEigenspace_toEnd hx hm
  /-
    🎉 no goals
  -/


lemma toEnd_pow_apply_mem {χ₁ χ₂ : H → R} {x : L} {m : M}
    (hx : x ∈ rootSpace H χ₁) (hm : m ∈ genWeightSpace M χ₂) (n) :
    (toEnd R L M x ^ n : Module.End R M) m ∈ genWeightSpace M (n • χ₁ + χ₂) := by
  induction n with
  | zero => simpa using hm
  | succ n IH =>
    simp only [pow_succ', LinearMap.mul_apply, toEnd_apply_apply,
      Nat.cast_add, Nat.cast_one, rootSpace]
    convert lie_mem_genWeightSpace_of_mem_genWeightSpace hx IH using 2
    rw [succ_nsmul, ← add_assoc, add_comm (n • _)]


/-- Auxiliary definition for `rootSpaceWeightSpaceProduct`,
which is close to the deterministic timeout limit.
-/
def rootSpaceWeightSpaceProductAux {χ₁ χ₂ χ₃ : H → R} (hχ : χ₁ + χ₂ = χ₃) :
    rootSpace H χ₁ →ₗ[R] genWeightSpace M χ₂ →ₗ[R] genWeightSpace M χ₃ where
  toFun x :=
    { toFun := fun m =>
        ⟨⁅(x : L), (m : M)⁆,
          hχ ▸ lie_mem_genWeightSpace_of_mem_genWeightSpace x.property m.property⟩
                                /-
                                  R : Type u_1
                                  L : Type u_2
                                  inst✝⁷ : CommRing R
                                  inst✝⁶ : LieRing L
                                  inst✝⁵ : LieAlgebra R L
                                  H : LieSubalgebra R L
                                  inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
                                  M : Type u_3
                                  inst✝³ : AddCommGroup M
                                  inst✝² : Module R M
                                  inst✝¹ : LieRingModule L M
                                  inst✝ : LieModule R L M
                                  χ₁ χ₂ χ₃ : (Subtype fun x => Membership.mem H x) → R
                                  hχ : Eq (HAdd.hAdd χ₁ χ₂) χ₃
                                  x : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₁) x
                                  m n : Subtype fun x => Membership.mem (LieModule.genWeightSpace M χ₂) x
                                  ⊢ Eq ((fun m => ⟨Bracket.bracket ↑x ↑m, ⋯⟩) (HAdd.hAdd m n)) (HAdd.hAdd ((fun  …
                                -/
      map_add' := fun m n => by simp only [LieSubmodule.coe_add, lie_add, AddMemClass.mk_add_mk]
                                /-
                                  🎉 no goals
                                -/
      map_smul' := fun t m => by
        /-
          R : Type u_1
          L : Type u_2
          inst✝⁷ : CommRing R
          inst✝⁶ : LieRing L
          inst✝⁵ : LieAlgebra R L
          H : LieSubalgebra R L
          inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
          M : Type u_3
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : LieRingModule L M
          inst✝ : LieModule R L M
          χ₁ χ₂ χ₃ : (Subtype fun x => Membership.mem H x) → R
          hχ : Eq (HAdd.hAdd χ₁ χ₂) χ₃
          x : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₁) x
          t : R
          m : Subtype fun x => Membership.mem (LieModule.genWeightSpace M χ₂) x
          ⊢ Eq ({ toFun := fun m => ⟨Bracket.bracket ↑x ↑m, ⋯⟩, map_add' := ⋯ }.toFun (H …
        -/
        dsimp only
        conv_lhs =>
          congr
          rw [LieSubmodule.coe_smul, lie_smul]
        /-
          R : Type u_1
          L : Type u_2
          inst✝⁷ : CommRing R
          inst✝⁶ : LieRing L
          inst✝⁵ : LieAlgebra R L
          H : LieSubalgebra R L
          inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
          M : Type u_3
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : LieRingModule L M
          inst✝ : LieModule R L M
          χ₁ χ₂ χ₃ : (Subtype fun x => Membership.mem H x) → R
          hχ : Eq (HAdd.hAdd χ₁ χ₂) χ₃
          x : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₁) x
          t : R
          m : Subtype fun x => Membership.mem (LieModule.genWeightSpace M χ₂) x
          ⊢ Eq ⟨HSMul.hSMul t (Bracket.bracket ↑x ↑m), ⋯⟩ (HSMul.hSMul ((RingHom.id R) t …
        -/
        rfl }
        /-
          🎉 no goals
        -/
  map_add' x y := by
    /-
      R : Type u_1
      L : Type u_2
      inst✝⁷ : CommRing R
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra R L
      H : LieSubalgebra R L
      inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
      M : Type u_3
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      χ₁ χ₂ χ₃ : (Subtype fun x => Membership.mem H x) → R
      hχ : Eq (HAdd.hAdd χ₁ χ₂) χ₃
      x y : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₁) x
      ⊢ Eq ((fun x => { toFun := fun m => ⟨Bracket.bracket ↑x ↑m, ⋯⟩, map_add' := ⋯, …
    -/
    ext m
    simp only [LieSubmodule.coe_add, add_lie, LinearMap.coe_mk, AddHom.coe_mk, LinearMap.add_apply,
      AddMemClass.mk_add_mk]
  map_smul' t x := by
    /-
      R : Type u_1
      L : Type u_2
      inst✝⁷ : CommRing R
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra R L
      H : LieSubalgebra R L
      inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
      M : Type u_3
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      χ₁ χ₂ χ₃ : (Subtype fun x => Membership.mem H x) → R
      hχ : Eq (HAdd.hAdd χ₁ χ₂) χ₃
      t : R
      x : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₁) x
      ⊢ Eq ({ toFun := fun x => { toFun := fun m => ⟨Bracket.bracket ↑x ↑m, ⋯⟩, map_ …
    -/
    simp only [RingHom.id_apply]
    /-
      R : Type u_1
      L : Type u_2
      inst✝⁷ : CommRing R
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra R L
      H : LieSubalgebra R L
      inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
      M : Type u_3
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      χ₁ χ₂ χ₃ : (Subtype fun x => Membership.mem H x) → R
      hχ : Eq (HAdd.hAdd χ₁ χ₂) χ₃
      t : R
      x : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₁) x
      ⊢ Eq { toFun := fun m => ⟨Bracket.bracket ↑(HSMul.hSMul t x) ↑m, ⋯⟩, map_add'  …
    -/
    ext m
    simp only [SetLike.val_smul, smul_lie, LinearMap.coe_mk, AddHom.coe_mk, LinearMap.smul_apply,
      SetLike.mk_smul_mk]


/-- Given a nilpotent Lie subalgebra `H ⊆ L` together with `χ₁ χ₂ : H → R`, there is a natural
`R`-bilinear product of root vectors and weight vectors, compatible with the actions of `H`. -/
def rootSpaceWeightSpaceProduct (χ₁ χ₂ χ₃ : H → R) (hχ : χ₁ + χ₂ = χ₃) :
    rootSpace H χ₁ ⊗[R] genWeightSpace M χ₂ →ₗ⁅R,H⁆ genWeightSpace M χ₃ :=
  liftLie R H (rootSpace H χ₁) (genWeightSpace M χ₂) (genWeightSpace M χ₃)
    { toLinearMap := rootSpaceWeightSpaceProductAux R L H M hχ
      map_lie' := fun {x y} => by
        /-
          R : Type u_1
          L : Type u_2
          inst✝⁷ : CommRing R
          inst✝⁶ : LieRing L
          inst✝⁵ : LieAlgebra R L
          H : LieSubalgebra R L
          inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
          M : Type u_3
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : LieRingModule L M
          inst✝ : LieModule R L M
          χ₁ χ₂ χ₃ : (Subtype fun x => Membership.mem H x) → R
          hχ : Eq (HAdd.hAdd χ₁ χ₂) χ₃
          x : Subtype fun x => Membership.mem H x
          y : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₁) x
          ⊢ Eq ((LieAlgebra.rootSpaceWeightSpaceProductAux R L H M hχ).toFun (Bracket.br …
        -/
        ext m
        /-
          case h.a
          R : Type u_1
          L : Type u_2
          inst✝⁷ : CommRing R
          inst✝⁶ : LieRing L
          inst✝⁵ : LieAlgebra R L
          H : LieSubalgebra R L
          inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
          M : Type u_3
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : LieRingModule L M
          inst✝ : LieModule R L M
          χ₁ χ₂ χ₃ : (Subtype fun x => Membership.mem H x) → R
          hχ : Eq (HAdd.hAdd χ₁ χ₂) χ₃
          x : Subtype fun x => Membership.mem H x
          y : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₁) x
          m : Subtype fun x => Membership.mem (LieModule.genWeightSpace M χ₂) x
          ⊢ Eq ↑(((LieAlgebra.rootSpaceWeightSpaceProductAux R L H M hχ).toFun (Bracket. …
        -/
        simp only [rootSpaceWeightSpaceProductAux]
        /-
          case h.a
          R : Type u_1
          L : Type u_2
          inst✝⁷ : CommRing R
          inst✝⁶ : LieRing L
          inst✝⁵ : LieAlgebra R L
          H : LieSubalgebra R L
          inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
          M : Type u_3
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : LieRingModule L M
          inst✝ : LieModule R L M
          χ₁ χ₂ χ₃ : (Subtype fun x => Membership.mem H x) → R
          hχ : Eq (HAdd.hAdd χ₁ χ₂) χ₃
          x : Subtype fun x => Membership.mem H x
          y : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₁) x
          m : Subtype fun x => Membership.mem (LieModule.genWeightSpace M χ₂) x
          ⊢ Eq ↑({ toFun := fun m => ⟨Bracket.bracket ↑(Bracket.bracket x y) ↑m, ⋯⟩, map …
        -/
        dsimp
        /-
          case h.a
          R : Type u_1
          L : Type u_2
          inst✝⁷ : CommRing R
          inst✝⁶ : LieRing L
          inst✝⁵ : LieAlgebra R L
          H : LieSubalgebra R L
          inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
          M : Type u_3
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : LieRingModule L M
          inst✝ : LieModule R L M
          χ₁ χ₂ χ₃ : (Subtype fun x => Membership.mem H x) → R
          hχ : Eq (HAdd.hAdd χ₁ χ₂) χ₃
          x : Subtype fun x => Membership.mem H x
          y : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₁) x
          m : Subtype fun x => Membership.mem (LieModule.genWeightSpace M χ₂) x
          ⊢ Eq (Bracket.bracket (Bracket.bracket ↑x ↑y) ↑m) (HSub.hSub (Bracket.bracket  …
        -/
        simp only [LieSubalgebra.coe_bracket_of_module, lie_lie] }
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_rootSpaceWeightSpaceProduct_tmul (χ₁ χ₂ χ₃ : H → R) (hχ : χ₁ + χ₂ = χ₃)
    (x : rootSpace H χ₁) (m : genWeightSpace M χ₂) :
    (rootSpaceWeightSpaceProduct R L H M χ₁ χ₂ χ₃ hχ (x ⊗ₜ m) : M) = ⁅(x : L), (m : M)⁆ := by
  simp only [rootSpaceWeightSpaceProduct, rootSpaceWeightSpaceProductAux, coe_liftLie_eq_lift_coe,
    AddHom.toFun_eq_coe, LinearMap.coe_toAddHom, lift_apply, LinearMap.coe_mk, AddHom.coe_mk,
    Submodule.coe_mk]


theorem mapsTo_toEnd_genWeightSpace_add_of_mem_rootSpace (α χ : H → R)
    {x : L} (hx : x ∈ rootSpace H α) :
    MapsTo (toEnd R L M x) (genWeightSpace M χ) (genWeightSpace M (α + χ)) := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    M : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    α χ : (Subtype fun x => Membership.mem H x) → R
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H α) x
    ⊢ Set.MapsTo ⇑((LieModule.toEnd R L M) x) ↑(LieModule.genWeightSpace M χ) ↑(Li …
  -/
  intro m hm
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    M : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    α χ : (Subtype fun x => Membership.mem H x) → R
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H α) x
    m : M
    hm : Membership.mem (↑(LieModule.genWeightSpace M χ)) m
    ⊢ Membership.mem (↑(LieModule.genWeightSpace M (HAdd.hAdd α χ))) (((LieModule. …
  -/
  let x' : rootSpace H α := ⟨x, hx⟩
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    M : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    α χ : (Subtype fun x => Membership.mem H x) → R
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H α) x
    m : M
    hm : Membership.mem (↑(LieModule.genWeightSpace M χ)) m
    x' : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H α) x := ⟨x, hx⟩
    ⊢ Membership.mem (↑(LieModule.genWeightSpace M (HAdd.hAdd α χ))) (((LieModule. …
  -/
  let m' : genWeightSpace M χ := ⟨m, hm⟩
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    M : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    α χ : (Subtype fun x => Membership.mem H x) → R
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H α) x
    m : M
    hm : Membership.mem (↑(LieModule.genWeightSpace M χ)) m
    x' : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H α) x := ⟨x, hx⟩
    m' : Subtype fun x => Membership.mem (LieModule.genWeightSpace M χ) x := ⟨m, hm⟩
    ⊢ Membership.mem (↑(LieModule.genWeightSpace M (HAdd.hAdd α χ))) (((LieModule. …
  -/
  exact (rootSpaceWeightSpaceProduct R L H M α χ (α + χ) rfl (x' ⊗ₜ m')).property
  /-
    🎉 no goals
  -/


/-- Given a nilpotent Lie subalgebra `H ⊆ L` together with `χ₁ χ₂ : H → R`, there is a natural
`R`-bilinear product of root vectors, compatible with the actions of `H`. -/
def rootSpaceProduct (χ₁ χ₂ χ₃ : H → R) (hχ : χ₁ + χ₂ = χ₃) :
    rootSpace H χ₁ ⊗[R] rootSpace H χ₂ →ₗ⁅R,H⁆ rootSpace H χ₃ :=
  rootSpaceWeightSpaceProduct R L H L χ₁ χ₂ χ₃ hχ


@[simp]
theorem rootSpaceProduct_def : rootSpaceProduct R L H = rootSpaceWeightSpaceProduct R L H L := rfl


theorem rootSpaceProduct_tmul
    (χ₁ χ₂ χ₃ : H → R) (hχ : χ₁ + χ₂ = χ₃) (x : rootSpace H χ₁) (y : rootSpace H χ₂) :
    (rootSpaceProduct R L H χ₁ χ₂ χ₃ hχ (x ⊗ₜ y) : L) = ⁅(x : L), (y : L)⁆ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    χ₁ χ₂ χ₃ : (Subtype fun x => Membership.mem H x) → R
    hχ : Eq (HAdd.hAdd χ₁ χ₂) χ₃
    x : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₁) x
    y : Subtype fun x => Membership.mem (LieAlgebra.rootSpace H χ₂) x
    ⊢ Eq (↑((LieAlgebra.rootSpaceProduct R L H χ₁ χ₂ χ₃ hχ) (TensorProduct.tmul R  …
  -/
  simp only [rootSpaceProduct_def, coe_rootSpaceWeightSpaceProduct_tmul]
  /-
    🎉 no goals
  -/


/-- Given a nilpotent Lie subalgebra `H ⊆ L`, the root space of the zero map `0 : H → R` is a Lie
subalgebra of `L`. -/
def zeroRootSubalgebra : LieSubalgebra R L :=
  { toSubmodule := (rootSpace H 0 : Submodule R L)
    lie_mem' := fun {x y hx hy} => by
      /-
        R : Type u_1
        L : Type u_2
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        H : LieSubalgebra R L
        inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
        M : Type u_3
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y : L
        hx : Membership.mem (↑(LieAlgebra.rootSpace H 0)).carrier x
        hy : Membership.mem (↑(LieAlgebra.rootSpace H 0)).carrier y
        ⊢ Membership.mem (↑(LieAlgebra.rootSpace H 0)).carrier (Bracket.bracket x y)
      -/
      let xy : rootSpace H 0 ⊗[R] rootSpace H 0 := ⟨x, hx⟩ ⊗ₜ ⟨y, hy⟩
      suffices (rootSpaceProduct R L H 0 0 0 (add_zero 0) xy : L) ∈ rootSpace H 0 by
        rwa [rootSpaceProduct_tmul, Subtype.coe_mk, Subtype.coe_mk] at this
      /-
        R : Type u_1
        L : Type u_2
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        H : LieSubalgebra R L
        inst✝⁴ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
        M : Type u_3
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y : L
        hx : Membership.mem (↑(LieAlgebra.rootSpace H 0)).carrier x
        hy : Membership.mem (↑(LieAlgebra.rootSpace H 0)).carrier y
        xy : TensorProduct R (Subtype fun x => Membership.mem (LieAlgebra.rootSpace H  …
        ⊢ Membership.mem (LieAlgebra.rootSpace H 0) ↑((LieAlgebra.rootSpaceProduct R L …
      -/
      exact (rootSpaceProduct R L H 0 0 0 (add_zero 0) xy).property }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_zeroRootSubalgebra : (zeroRootSubalgebra R L H : Submodule R L) = rootSpace H 0 := rfl


theorem mem_zeroRootSubalgebra (x : L) :
    x ∈ zeroRootSubalgebra R L H ↔ ∀ y : H, ∃ k : ℕ, (toEnd R H L y ^ k) x = 0 := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    ⊢ Iff (Membership.mem (LieAlgebra.zeroRootSubalgebra R L H) x) (∀ (y : Subtype …
  -/
  change x ∈ rootSpace H 0 ↔ _
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    ⊢ Iff (Membership.mem (LieAlgebra.rootSpace H 0) x) (∀ (y : Subtype fun x => M …
  -/
  simp only [mem_genWeightSpace, Pi.zero_apply, zero_smul, sub_zero]
  /-
    🎉 no goals
  -/


theorem toLieSubmodule_le_rootSpace_zero : H.toLieSubmodule ≤ rootSpace H 0 := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    ⊢ LE.le H.toLieSubmodule (LieAlgebra.rootSpace H 0)
  -/
  intro x hx
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H.toLieSubmodule x
    ⊢ Membership.mem (LieAlgebra.rootSpace H 0) x
  -/
  simp only [LieSubalgebra.mem_toLieSubmodule] at hx
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H x
    ⊢ Membership.mem (LieAlgebra.rootSpace H 0) x
  -/
  simp only [mem_genWeightSpace, Pi.zero_apply, sub_zero, zero_smul]
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H x
    ⊢ ∀ (x_1 : Subtype fun x => Membership.mem H x), Exists fun k => Eq ((HPow.hPo …
  -/
  intro y
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H x
    y : Subtype fun x => Membership.mem H x
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Members …
  -/
  obtain ⟨k, hk⟩ := (inferInstance : IsNilpotent R H)
  /-
    case mk.intro
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H x
    y : Subtype fun x => Membership.mem H x
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R (Subtype fun x => Membership.mem H x)  …
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Members …
  -/
  use k
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H x
    y : Subtype fun x => Membership.mem H x
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R (Subtype fun x => Membership.mem H x)  …
    ⊢ Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Membership.mem H x) L)  …
  -/
  let f : Module.End R H := toEnd R H H y
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H x
    y : Subtype fun x => Membership.mem H x
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R (Subtype fun x => Membership.mem H x)  …
    f : Module.End R (Subtype fun x => Membership.mem H x) := (LieModule.toEnd R ( …
    ⊢ Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Membership.mem H x) L)  …
  -/
  let g : Module.End R L := toEnd R H L y
  have hfg : g.comp (H : Submodule R L).subtype = (H : Submodule R L).subtype.comp f := by
    ext z
    simp only [toEnd_apply_apply, Submodule.subtype_apply,
      LieSubalgebra.coe_bracket_of_module, LieSubalgebra.coe_bracket, Function.comp_apply,
      LinearMap.coe_comp]
    rfl
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H x
    y : Subtype fun x => Membership.mem H x
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R (Subtype fun x => Membership.mem H x)  …
    f : Module.End R (Subtype fun x => Membership.mem H x) := (LieModule.toEnd R ( …
    g : Module.End R L := (LieModule.toEnd R (Subtype fun x => Membership.mem H x) …
    hfg : Eq (LinearMap.comp g H.subtype) (H.subtype.comp f)
    ⊢ Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Membership.mem H x) L)  …
  -/
  change (g ^ k).comp (H : Submodule R L).subtype ⟨x, hx⟩ = 0
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H x
    y : Subtype fun x => Membership.mem H x
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R (Subtype fun x => Membership.mem H x)  …
    f : Module.End R (Subtype fun x => Membership.mem H x) := (LieModule.toEnd R ( …
    g : Module.End R L := (LieModule.toEnd R (Subtype fun x => Membership.mem H x) …
    hfg : Eq (LinearMap.comp g H.subtype) (H.subtype.comp f)
    ⊢ Eq ((LinearMap.comp (HPow.hPow g k) H.subtype) ⟨x, hx⟩) 0
  -/
  rw [LinearMap.commute_pow_left_of_commute hfg k]
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H x
    y : Subtype fun x => Membership.mem H x
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R (Subtype fun x => Membership.mem H x)  …
    f : Module.End R (Subtype fun x => Membership.mem H x) := (LieModule.toEnd R ( …
    g : Module.End R L := (LieModule.toEnd R (Subtype fun x => Membership.mem H x) …
    hfg : Eq (LinearMap.comp g H.subtype) (H.subtype.comp f)
    ⊢ Eq ((H.subtype.comp (HPow.hPow f k)) ⟨x, hx⟩) 0
  -/
  have h := iterate_toEnd_mem_lowerCentralSeries R H H y ⟨x, hx⟩ k
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H x
    y : Subtype fun x => Membership.mem H x
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R (Subtype fun x => Membership.mem H x)  …
    f : Module.End R (Subtype fun x => Membership.mem H x) := (LieModule.toEnd R ( …
    g : Module.End R L := (LieModule.toEnd R (Subtype fun x => Membership.mem H x) …
    hfg : Eq (LinearMap.comp g H.subtype) (H.subtype.comp f)
    h : Membership.mem (LieModule.lowerCentralSeries R (Subtype fun x => Membershi …
    ⊢ Eq ((H.subtype.comp (HPow.hPow f k)) ⟨x, hx⟩) 0
  -/
  rw [hk, LieSubmodule.mem_bot] at h
  simp only [Submodule.subtype_apply, Function.comp_apply, LinearMap.pow_apply, LinearMap.coe_comp,
    Submodule.coe_eq_zero]
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem H x
    y : Subtype fun x => Membership.mem H x
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R (Subtype fun x => Membership.mem H x)  …
    f : Module.End R (Subtype fun x => Membership.mem H x) := (LieModule.toEnd R ( …
    g : Module.End R L := (LieModule.toEnd R (Subtype fun x => Membership.mem H x) …
    hfg : Eq (LinearMap.comp g H.subtype) (H.subtype.comp f)
    h : Eq (Nat.iterate (⇑((LieModule.toEnd R (Subtype fun x => Membership.mem H x …
    ⊢ Eq (Nat.iterate (⇑f) k ⟨x, hx⟩) 0
  -/
  exact h
  /-
    🎉 no goals
  -/


/-- This enables the instance `Zero (Weight R H L)`. -/
instance [Nontrivial H] : Nontrivial (genWeightSpace L (0 : H → R)) := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝⁵ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : Nontrivial (Subtype fun x => Membership.mem H x)
    ⊢ Nontrivial (Subtype fun x => Membership.mem (LieModule.genWeightSpace L 0) x)
  -/
  obtain ⟨⟨x, hx⟩, ⟨y, hy⟩, e⟩ := exists_pair_ne H
  exact ⟨⟨x, toLieSubmodule_le_rootSpace_zero R L H hx⟩,
    ⟨y, toLieSubmodule_le_rootSpace_zero R L H hy⟩, by simpa using e⟩


theorem le_zeroRootSubalgebra : H ≤ zeroRootSubalgebra R L H := by
  rw [← LieSubalgebra.toSubmodule_le_toSubmodule, ← H.coe_toLieSubmodule,
    coe_zeroRootSubalgebra, LieSubmodule.toSubmodule_le_toSubmodule]
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    ⊢ LE.le H.toLieSubmodule (LieAlgebra.rootSpace H 0)
  -/
  exact toLieSubmodule_le_rootSpace_zero R L H
  /-
    🎉 no goals
  -/


@[simp]
theorem zeroRootSubalgebra_normalizer_eq_self :
    (zeroRootSubalgebra R L H).normalizer = zeroRootSubalgebra R L H := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    ⊢ Eq (LieAlgebra.zeroRootSubalgebra R L H).normalizer (LieAlgebra.zeroRootSuba …
  -/
  refine le_antisymm ?_ (LieSubalgebra.le_normalizer _)
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    ⊢ LE.le (LieAlgebra.zeroRootSubalgebra R L H).normalizer (LieAlgebra.zeroRootS …
  -/
  intro x hx
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem (LieAlgebra.zeroRootSubalgebra R L H).normalizer x
    ⊢ Membership.mem (LieAlgebra.zeroRootSubalgebra R L H) x
  -/
  rw [LieSubalgebra.mem_normalizer_iff] at hx
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : ∀ (y : L), Membership.mem (LieAlgebra.zeroRootSubalgebra R L H) y → Membe …
    ⊢ Membership.mem (LieAlgebra.zeroRootSubalgebra R L H) x
  -/
  rw [mem_zeroRootSubalgebra]
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : ∀ (y : L), Membership.mem (LieAlgebra.zeroRootSubalgebra R L H) y → Membe …
    ⊢ ∀ (y : Subtype fun x => Membership.mem H x), Exists fun k => Eq ((HPow.hPow  …
  -/
  rintro ⟨y, hy⟩
  /-
    case mk
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : ∀ (y : L), Membership.mem (LieAlgebra.zeroRootSubalgebra R L H) y → Membe …
    y : L
    hy : Membership.mem H y
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Members …
  -/
  specialize hx y (le_zeroRootSubalgebra R L H hy)
  /-
    case mk
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x y : L
    hy : Membership.mem H y
    hx : Membership.mem (LieAlgebra.zeroRootSubalgebra R L H) (Bracket.bracket x y)
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Members …
  -/
  rw [mem_zeroRootSubalgebra] at hx
  /-
    case mk
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x y : L
    hy : Membership.mem H y
    hx : ∀ (y_1 : Subtype fun x => Membership.mem H x), Exists fun k => Eq ((HPow. …
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Members …
  -/
  obtain ⟨k, hk⟩ := hx ⟨y, hy⟩
  /-
    case mk.intro
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x y : L
    hy : Membership.mem H y
    hx : ∀ (y_1 : Subtype fun x => Membership.mem H x), Exists fun k => Eq ((HPow. …
    k : Nat
    hk : Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Membership.mem H x)  …
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Members …
  -/
  rw [← lie_skew, LinearMap.map_neg, neg_eq_zero] at hk
  /-
    case mk.intro
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x y : L
    hy : Membership.mem H y
    hx : ∀ (y_1 : Subtype fun x => Membership.mem H x), Exists fun k => Eq ((HPow. …
    k : Nat
    hk : Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Membership.mem H x)  …
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R (Subtype fun x => Members …
  -/
  use k + 1
  rw [LinearMap.iterate_succ, LinearMap.coe_comp, Function.comp_apply, toEnd_apply_apply,
    LieSubalgebra.coe_bracket_of_module, Submodule.coe_mk, hk]


/-- If the zero root subalgebra of a nilpotent Lie subalgebra `H` is just `H` then `H` is a Cartan
subalgebra.

When `L` is Noetherian, it follows from Engel's theorem that the converse holds. See
`LieAlgebra.zeroRootSubalgebra_eq_iff_is_cartan` -/
theorem is_cartan_of_zeroRootSubalgebra_eq (h : zeroRootSubalgebra R L H = H) :
    H.IsCartanSubalgebra :=
  { nilpotent := inferInstance
                           /-
                             R : Type u_1
                             L : Type u_2
                             inst✝³ : CommRing R
                             inst✝² : LieRing L
                             inst✝¹ : LieAlgebra R L
                             H : LieSubalgebra R L
                             inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
                             h : Eq (LieAlgebra.zeroRootSubalgebra R L H) H
                             ⊢ Eq H.normalizer H
                           -/
    self_normalizing := by rw [← h]; exact zeroRootSubalgebra_normalizer_eq_self R L H }
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem zeroRootSubalgebra_eq_of_is_cartan (H : LieSubalgebra R L) [H.IsCartanSubalgebra]
    [IsNoetherian R L] : zeroRootSubalgebra R L H = H := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    ⊢ Eq (LieAlgebra.zeroRootSubalgebra R L H) H
  -/
  refine le_antisymm ?_ (le_zeroRootSubalgebra R L H)
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    ⊢ LE.le (LieAlgebra.zeroRootSubalgebra R L H) H
  -/
  suffices rootSpace H 0 ≤ H.toLieSubmodule by exact fun x hx => this hx
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    ⊢ LE.le (LieAlgebra.rootSpace H 0) H.toLieSubmodule
  -/
  obtain ⟨k, hk⟩ := (rootSpace H 0).isNilpotent_iff_exists_self_le_ucs.mp (by infer_instance)
  /-
    case intro
    R : Type u_1
    L : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    k : Nat
    hk : LE.le (LieAlgebra.rootSpace H 0) (LieSubmodule.ucs k Bot.bot)
    ⊢ LE.le (LieAlgebra.rootSpace H 0) H.toLieSubmodule
  -/
  exact hk.trans (LieSubmodule.ucs_le_of_normalizer_eq_self (by simp) k)
  /-
    🎉 no goals
  -/


theorem zeroRootSubalgebra_eq_iff_is_cartan [IsNoetherian R L] :
    zeroRootSubalgebra R L H = H ↔ H.IsCartanSubalgebra :=
                                                /-
                                                  R : Type u_1
                                                  L : Type u_2
                                                  inst✝⁴ : CommRing R
                                                  inst✝³ : LieRing L
                                                  inst✝² : LieAlgebra R L
                                                  H : LieSubalgebra R L
                                                  inst✝¹ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
                                                  inst✝ : IsNoetherian R L
                                                  ⊢ H.IsCartanSubalgebra → Eq (LieAlgebra.zeroRootSubalgebra R L H) H
                                                -/
  ⟨is_cartan_of_zeroRootSubalgebra_eq R L H, by intros; simp⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem rootSpace_zero_eq (H : LieSubalgebra R L) [H.IsCartanSubalgebra] [IsNoetherian R L] :
    rootSpace H 0 = H.toLieSubmodule := by
  rw [← LieSubmodule.toSubmodule_inj, ← coe_zeroRootSubalgebra,
    zeroRootSubalgebra_eq_of_is_cartan R L H, LieSubalgebra.coe_toLieSubmodule]


/-- Given a root `α` relative to a Cartan subalgebra `H`, this is the span of all products of
an element of the `α` root space and an element of the `-α` root space. Informally it is often
denoted `⁅H(α), H(-α)⁆`.

If the Killing form is non-degenerate and the coefficients are a perfect field, this space is
one-dimensional. See `LieAlgebra.IsKilling.coe_corootSpace_eq_span_singleton` and
`LieAlgebra.IsKilling.coe_corootSpace_eq_span_singleton'`.

Note that the name "coroot space" is not standard as this space does not seem to have a name in the
informal literature. -/
def corootSpace : LieIdeal R H :=
  LieModuleHom.range <| ((rootSpace H 0).incl.comp <|
    rootSpaceProduct R L H α (-α) 0 (add_neg_cancel α)).codRestrict H.toLieSubmodule (by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝⁶ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    α : (Subtype fun x => Membership.mem H x) → R
    ⊢ ∀ (m : TensorProduct R (Subtype fun x => Membership.mem (LieAlgebra.rootSpac …
  -/
  rw [← rootSpace_zero_eq]
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝⁶ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    α : (Subtype fun x => Membership.mem H x) → R
    ⊢ ∀ (m : TensorProduct R (Subtype fun x => Membership.mem (LieAlgebra.rootSpac …
  -/
  exact fun p ↦ (rootSpaceProduct R L H α (-α) 0 (add_neg_cancel α) p).property)
  /-
    🎉 no goals
  -/


lemma mem_corootSpace {x : H} :
    x ∈ corootSpace α ↔
    (x : L) ∈ Submodule.span R {⁅y, z⁆ | (y ∈ rootSpace H α) (z ∈ rootSpace H (-α))} := by
  have : x ∈ corootSpace α ↔
      (x : L) ∈ LieSubmodule.map H.toLieSubmodule.incl (corootSpace α) := by
    rw [corootSpace]
    simp only [rootSpaceProduct_def, LieModuleHom.mem_range, LieSubmodule.mem_map,
      LieSubmodule.incl_apply, SetLike.coe_eq_coe, exists_eq_right]
    rfl
  simp_rw [this, corootSpace, ← LieModuleHom.map_top, ← LieSubmodule.mem_toSubmodule,
    LieSubmodule.toSubmodule_map, LieSubmodule.top_toSubmodule, ← TensorProduct.span_tmul_eq_top,
    LinearMap.map_span, Set.image, Set.mem_setOf_eq, exists_exists_exists_and_eq]
  change (x : L) ∈ Submodule.span R
    {x | ∃ (a : rootSpace H α) (b : rootSpace H (-α)), ⁅(a : L), (b : L)⁆ = x} ↔ _
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝² : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    α : (Subtype fun x => Membership.mem H x) → R
    x : Subtype fun x => Membership.mem H x
    this : Iff (Membership.mem (LieAlgebra.corootSpace α) x) (Membership.mem (LieS …
    ⊢ Iff (Membership.mem (Submodule.span R (setOf fun x => Exists fun a => Exists …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma mem_corootSpace' {x : H} :
    x ∈ corootSpace α ↔
    x ∈ Submodule.span R ({⁅y, z⁆ | (y ∈ rootSpace H α) (z ∈ rootSpace H (-α))} : Set H) := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝² : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    α : (Subtype fun x => Membership.mem H x) → R
    x : Subtype fun x => Membership.mem H x
    ⊢ Iff (Membership.mem (LieAlgebra.corootSpace α) x) (Membership.mem (Submodule …
  -/
  set s : Set H := ({⁅y, z⁆ | (y ∈ rootSpace H α) (z ∈ rootSpace H (-α))} : Set H)
  suffices H.subtype '' s = {⁅y, z⁆ | (y ∈ rootSpace H α) (z ∈ rootSpace H (-α))} by
    obtain ⟨x, hx⟩ := x
    erw [← (H : Submodule R L).injective_subtype.mem_set_image (s := Submodule.span R s)]
    rw [mem_image]
    simp_rw [SetLike.mem_coe]
    rw [← Submodule.mem_map, Submodule.coe_subtype, Submodule.map_span, mem_corootSpace, ← this]
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝² : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    α : (Subtype fun x => Membership.mem H x) → R
    x : Subtype fun x => Membership.mem H x
    s : Set (Subtype fun x => Membership.mem H x) := setOf fun x => Exists fun y = …
    ⊢ Eq (Set.image (⇑H.subtype) s) (setOf fun x => Exists fun y => And (Membershi …
  -/
  ext u
  simp only [Submodule.coe_subtype, mem_image, Subtype.exists, LieSubalgebra.mem_toSubmodule,
    exists_and_right, exists_eq_right, mem_setOf_eq, s]
  refine ⟨fun ⟨_, y, hy, z, hz, hyz⟩ ↦ ⟨y, hy, z, hz, hyz⟩,
    fun ⟨y, hy, z, hz, hyz⟩ ↦ ⟨?_, y, hy, z, hz, hyz⟩⟩
  convert
    (rootSpaceProduct R L H α (-α) 0 (add_neg_cancel α) (⟨y, hy⟩ ⊗ₜ[R] ⟨z, hz⟩)).property using 0
  /-
    case a
    R : Type u_1
    L : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    H : LieSubalgebra R L
    inst✝² : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    α : (Subtype fun x => Membership.mem H x) → R
    x : Subtype fun x => Membership.mem H x
    s : Set (Subtype fun x => Membership.mem H x) := setOf fun x => Exists fun y = …
    u : L
    x✝ : Exists fun y => And (Membership.mem (LieAlgebra.rootSpace H α) y) (Exists …
    y : L
    hy : Membership.mem (LieAlgebra.rootSpace H α) y
    z : L
    hz : Membership.mem (LieAlgebra.rootSpace H (Neg.neg α)) z
    hyz : Eq (Bracket.bracket y z) u
    ⊢ Iff (Membership.mem H u) (Membership.mem (LieAlgebra.rootSpace H 0) ↑((LieAl …
  -/
  simp [hyz]
  /-
    🎉 no goals
  -/


