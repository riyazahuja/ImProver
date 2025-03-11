/-- The character of a representation `V : FDRep k G` is the function associating to `g : G` the
trace of the linear map `V.ρ g`. -/
def character (V : FDRep k G) (g : G) :=
  LinearMap.trace k V (V.ρ g)


theorem char_mul_comm (V : FDRep k G) (g : G) (h : G) :
                                                    /-
                                                      k : Type u
                                                      inst✝¹ : Field k
                                                      G : Type u
                                                      inst✝ : Monoid G
                                                      V : FDRep k G
                                                      g h : G
                                                      ⊢ Eq (V.character (HMul.hMul h g)) (V.character (HMul.hMul g h))
                                                    -/
    V.character (h * g) = V.character (g * h) := by simp only [trace_mul_comm, character, map_mul]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem char_one (V : FDRep k G) : V.character 1 = Module.finrank k V := by
  /-
    k : Type u
    inst✝¹ : Field k
    G : Type u
    inst✝ : Monoid G
    V : FDRep k G
    ⊢ Eq (V.character 1) ↑(Module.finrank k (CoeSort.coe V))
  -/
  simp only [character, map_one, trace_one]
  /-
    🎉 no goals
  -/


/-- The character is multiplicative under the tensor product. -/
@[simp]
theorem char_tensor (V W : FDRep k G) : (V ⊗ W).character = V.character * W.character := by
  /-
    k : Type u
    inst✝¹ : Field k
    G : Type u
    inst✝ : Monoid G
    V W : FDRep k G
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorObj V W).character (HMul.hMu …
  -/
  ext g; convert trace_tensorProduct' (V.ρ g) (W.ρ g)
         /-
           🎉 no goals
         -/


/-- The character of isomorphic representations is the same. -/
theorem char_iso {V W : FDRep k G} (i : V ≅ W) : V.character = W.character := by
  /-
    k : Type u
    inst✝¹ : Field k
    G : Type u
    inst✝ : Monoid G
    V W : FDRep k G
    i : CategoryTheory.Iso V W
    ⊢ Eq V.character W.character
  -/
  ext g
  /-
    case h
    k : Type u
    inst✝¹ : Field k
    G : Type u
    inst✝ : Monoid G
    V W : FDRep k G
    i : CategoryTheory.Iso V W
    g : G
    ⊢ Eq (V.character g) (W.character g)
  -/
  simp only [character, FDRep.Iso.conj_ρ i]
  /-
    case h
    k : Type u
    inst✝¹ : Field k
    G : Type u
    inst✝ : Monoid G
    V W : FDRep k G
    i : CategoryTheory.Iso V W
    g : G
    ⊢ Eq ((LinearMap.trace k (CoeSort.coe V)) (V.ρ g)) ((LinearMap.trace k (CoeSor …
  -/
  exact (trace_conj' (V.ρ g) _).symm
  /-
    🎉 no goals
  -/


/-- The character of a representation is constant on conjugacy classes. -/
@[simp]
theorem char_conj (V : FDRep k G) (g : G) (h : G) : V.character (h * g * h⁻¹) = V.character g := by
  /-
    k : Type u
    inst✝¹ : Field k
    G : Type u
    inst✝ : Group G
    V : FDRep k G
    g h : G
    ⊢ Eq (V.character (HMul.hMul (HMul.hMul h g) (Inv.inv h))) (V.character g)
  -/
  rw [char_mul_comm, inv_mul_cancel_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem char_dual (V : FDRep k G) (g : G) : (of (dual V.ρ)).character g = V.character g⁻¹ :=
  trace_transpose' (V.ρ g⁻¹)


@[simp]
theorem char_linHom (V W : FDRep k G) (g : G) :
    (of (linHom V.ρ W.ρ)).character g = V.character g⁻¹ * W.character g := by
  /-
    k : Type u
    inst✝¹ : Field k
    G : Type u
    inst✝ : Group G
    V W : FDRep k G
    g : G
    ⊢ Eq ((FDRep.of (Representation.linHom V.ρ W.ρ)).character g) (HMul.hMul (V.ch …
  -/
  rw [← char_iso (dualTensorIsoLinHom _ _), char_tensor, Pi.mul_apply, char_dual]
  /-
    🎉 no goals
  -/


theorem average_char_eq_finrank_invariants (V : FDRep k G) :
    ⅟ (Fintype.card G : k) • ∑ g : G, V.character g = finrank k (invariants V.ρ) := by
  /-
    k : Type u
    inst✝³ : Field k
    G : Type u
    inst✝² : Group G
    inst✝¹ : Fintype G
    inst✝ : Invertible ↑(Fintype.card G)
    V : FDRep k G
    ⊢ Eq (HSMul.hSMul (Invertible.invOf ↑(Fintype.card G)) (Finset.univ.sum fun g  …
  -/
  rw [← (isProj_averageMap V.ρ).trace]
  /-
    k : Type u
    inst✝³ : Field k
    G : Type u
    inst✝² : Group G
    inst✝¹ : Fintype G
    inst✝ : Invertible ↑(Fintype.card G)
    V : FDRep k G
    ⊢ Eq (HSMul.hSMul (Invertible.invOf ↑(Fintype.card G)) (Finset.univ.sum fun g  …
  -/
  simp [character, GroupAlgebra.average, _root_.map_sum]
  /-
    🎉 no goals
  -/


/-- Orthogonality of characters for irreducible representations of finite group over an
algebraically closed field whose characteristic doesn't divide the order of the group. -/
theorem char_orthonormal (V W : FDRep k G) [Simple V] [Simple W] :
    ⅟ (Fintype.card G : k) • ∑ g : G, V.character g * W.character g⁻¹ =
      if Nonempty (V ≅ W) then ↑1 else ↑0 := by
  -- First, we can rewrite the summand `V.character g * W.character g⁻¹` as the character
  -- of the representation `V ⊗ W* ≅ Hom(W, V)` applied to `g`.
  -- Porting note: Originally `conv in V.character _ * W.character _ =>`
  conv_lhs =>
    enter [2, 2, g]
    rw [mul_comm, ← char_dual, ← Pi.mul_apply, ← char_tensor]
    rw [char_iso (FDRep.dualTensorIsoLinHom W.ρ V)]
  -- The average over the group of the character of a representation equals the dimension of the
  -- space of invariants.
  rw [average_char_eq_finrank_invariants, ← FDRep.endMulEquiv_comp_ρ (of _),
      FDRep.of_ρ (linHom W.ρ V.ρ)]
  -- The space of invariants of `Hom(W, V)` is the subspace of `G`-equivariant linear maps,
  -- `Hom_G(W, V)`.
  /-
    k : Type u
    inst✝⁵ : Field k
    G : Grp
    inst✝⁴ : IsAlgClosed k
    inst✝³ : Fintype ↑G
    inst✝² : Invertible ↑(Fintype.card ↑G)
    V W : FDRep k ↑G
    inst✝¹ : CategoryTheory.Simple V
    inst✝ : CategoryTheory.Simple W
    ⊢ Eq (↑(Module.finrank k (Subtype fun x => Membership.mem (Representation.inva …
  -/
  erw [(linHom.invariantsEquivFDRepHom W V).finrank_eq] -- Porting note: Changed `rw` to `erw`
  -- By Schur's Lemma, the dimension of `Hom_G(W, V)` is `1` is `V ≅ W` and `0` otherwise.
  /-
    k : Type u
    inst✝⁵ : Field k
    G : Grp
    inst✝⁴ : IsAlgClosed k
    inst✝³ : Fintype ↑G
    inst✝² : Invertible ↑(Fintype.card ↑G)
    V W : FDRep k ↑G
    inst✝¹ : CategoryTheory.Simple V
    inst✝ : CategoryTheory.Simple W
    ⊢ Eq (↑(Module.finrank k (Quiver.Hom W V))) (ite (Nonempty (CategoryTheory.Iso …
  -/
  rw_mod_cast [finrank_hom_simple_simple W V, Iso.nonempty_iso_symm]
  /-
    🎉 no goals
  -/


