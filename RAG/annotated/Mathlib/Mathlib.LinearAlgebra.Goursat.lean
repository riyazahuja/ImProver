variable (L) in
/-- For `L` a submodule of `M × N`, `L.goursatFst` is the kernel of the projection map `L → N`,
considered as a submodule of `M`.

This is the first submodule appearing in Goursat's lemma. See `Subgroup.goursat`. -/
def goursatFst : Submodule R M :=
  ((LinearMap.snd R M N).comp L.subtype).ker.map ((LinearMap.fst R M N).comp L.subtype)



variable (L) in
/-- For `L` a subgroup of `M × N`, `L.goursatSnd` is the kernel of the projection map `L → M`,
considered as a subgroup of `N`.

This is the second subgroup appearing in Goursat's lemma. See `Subgroup.goursat`. -/
def goursatSnd : Submodule R N :=
  ((LinearMap.fst R M N).comp L.subtype).ker.map ((LinearMap.snd R M N).comp L.subtype)


lemma goursatFst_toAddSubgroup :
    (goursatFst L).toAddSubgroup = L.toAddSubgroup.goursatFst := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    ⊢ Eq L.goursatFst.toAddSubgroup L.toAddSubgroup.goursatFst
  -/
  ext x
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    x : M
    ⊢ Iff (Membership.mem L.goursatFst.toAddSubgroup x) (Membership.mem L.toAddSub …
  -/
  simp [mem_toAddSubgroup, goursatFst, AddSubgroup.mem_goursatFst]
  /-
    🎉 no goals
  -/


lemma goursatSnd_toAddSubgroup :
    (goursatSnd L).toAddSubgroup = L.toAddSubgroup.goursatSnd := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    ⊢ Eq L.goursatSnd.toAddSubgroup L.toAddSubgroup.goursatSnd
  -/
  ext x
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    x : N
    ⊢ Iff (Membership.mem L.goursatSnd.toAddSubgroup x) (Membership.mem L.toAddSub …
  -/
  simp [mem_toAddSubgroup, goursatSnd, AddSubgroup.mem_goursatSnd]
  /-
    🎉 no goals
  -/


variable (L) in
lemma goursatFst_prod_goursatSnd_le : L.goursatFst.prod L.goursatSnd ≤ L := by
  simpa only [← toAddSubgroup_le, goursatFst_toAddSubgroup, goursatSnd_toAddSubgroup]
    using L.toAddSubgroup.goursatFst_prod_goursatSnd_le


include hL₁ hL₂ in
/-- **Goursat's lemma** for a submodule of a product with surjective projections.

If `L` is a submodule of `M × N` which projects fully on both factors, then there exist submodules
`M' ≤ M` and `N' ≤ N` such that `M' × N' ≤ L` and the image of `L` in `(M ⧸ M') × (N ⧸ N')` is the
graph of an isomorphism of `R`-modules `(M ⧸ M') ≃ (N ⧸ N')`.

`M` and `N` can be explicitly constructed as `L.goursatFst` and `L.goursatSnd` respectively. -/
lemma goursat_surjective : ∃ e : (M ⧸ L.goursatFst) ≃ₗ[R] N ⧸ L.goursatSnd,
    LinearMap.range ((L.goursatFst.mkQ.prodMap L.goursatSnd.mkQ).comp L.subtype) = e.graph := by
  -- apply add-group result
  obtain ⟨(e : M ⧸ L.goursatFst ≃+ N ⧸ L.goursatSnd), he⟩ :=
    L.toAddSubgroup.goursat_surjective hL₁ hL₂
  -- check R-linearity of the map
  have (r : R) (x : M ⧸ L.goursatFst) : e (r • x) = r • e x := by
    show (r • x, r • e x) ∈ e.toAddMonoidHom.graph
    rw [← he, ← Prod.smul_mk]
    have : (x, e x) ∈ e.toAddMonoidHom.graph := rfl
    rw [← he, AddMonoidHom.mem_range] at this
    rcases this with ⟨⟨l, hl⟩, hl'⟩
    use ⟨r • l, L.smul_mem r hl⟩
    rw [← hl']
    rfl
  -- define the map as an R-linear equiv
  /-
    case intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    hL₁ : Function.Surjective (Function.comp Prod.fst ⇑L.subtype)
    hL₂ : Function.Surjective (Function.comp Prod.snd ⇑L.subtype)
    e : AddEquiv (HasQuotient.Quotient M L.goursatFst) (HasQuotient.Quotient N L.g …
    he : Eq (((QuotientAddGroup.mk' L.toAddSubgroup.goursatFst).prodMap (QuotientA …
    this : ∀ (r : R) (x : HasQuotient.Quotient M L.goursatFst), Eq (e (HSMul.hSMul …
    ⊢ Exists fun e => Eq (LinearMap.range ((L.goursatFst.mkQ.prodMap L.goursatSnd. …
  -/
  use { e with map_smul' := this }
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    hL₁ : Function.Surjective (Function.comp Prod.fst ⇑L.subtype)
    hL₂ : Function.Surjective (Function.comp Prod.snd ⇑L.subtype)
    e : AddEquiv (HasQuotient.Quotient M L.goursatFst) (HasQuotient.Quotient N L.g …
    he : Eq (((QuotientAddGroup.mk' L.toAddSubgroup.goursatFst).prodMap (QuotientA …
    this : ∀ (r : R) (x : HasQuotient.Quotient M L.goursatFst), Eq (e (HSMul.hSMul …
    ⊢ Eq (LinearMap.range ((L.goursatFst.mkQ.prodMap L.goursatSnd.mkQ).comp L.subt …
  -/
  rw [← toAddSubgroup_injective.eq_iff]
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    hL₁ : Function.Surjective (Function.comp Prod.fst ⇑L.subtype)
    hL₂ : Function.Surjective (Function.comp Prod.snd ⇑L.subtype)
    e : AddEquiv (HasQuotient.Quotient M L.goursatFst) (HasQuotient.Quotient N L.g …
    he : Eq (((QuotientAddGroup.mk' L.toAddSubgroup.goursatFst).prodMap (QuotientA …
    this : ∀ (r : R) (x : HasQuotient.Quotient M L.goursatFst), Eq (e (HSMul.hSMul …
    ⊢ Eq (LinearMap.range ((L.goursatFst.mkQ.prodMap L.goursatSnd.mkQ).comp L.subt …
  -/
  convert he using 1
  /-
    case h.e'_3.h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    hL₁ : Function.Surjective (Function.comp Prod.fst ⇑L.subtype)
    hL₂ : Function.Surjective (Function.comp Prod.snd ⇑L.subtype)
    e : AddEquiv (HasQuotient.Quotient M L.goursatFst) (HasQuotient.Quotient N L.g …
    he : Eq (((QuotientAddGroup.mk' L.toAddSubgroup.goursatFst).prodMap (QuotientA …
    this : ∀ (r : R) (x : HasQuotient.Quotient M L.goursatFst), Eq (e (HSMul.hSMul …
    e_1✝ : Eq (AddSubgroup (Prod (HasQuotient.Quotient M L.goursatFst) (HasQuotien …
    ⊢ Eq (↑{ toFun := e.toFun, map_add' := ⋯, map_smul' := this, invFun := e.invFu …
  -/
  ext v
  /-
    case h.e'_3.h.h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    hL₁ : Function.Surjective (Function.comp Prod.fst ⇑L.subtype)
    hL₂ : Function.Surjective (Function.comp Prod.snd ⇑L.subtype)
    e : AddEquiv (HasQuotient.Quotient M L.goursatFst) (HasQuotient.Quotient N L.g …
    he : Eq (((QuotientAddGroup.mk' L.toAddSubgroup.goursatFst).prodMap (QuotientA …
    this : ∀ (r : R) (x : HasQuotient.Quotient M L.goursatFst), Eq (e (HSMul.hSMul …
    e_1✝ : Eq (AddSubgroup (Prod (HasQuotient.Quotient M L.goursatFst) (HasQuotien …
    v : Prod (HasQuotient.Quotient M L.goursatFst) (HasQuotient.Quotient N L.gours …
    ⊢ Iff (Membership.mem (↑{ toFun := e.toFun, map_add' := ⋯, map_smul' := this,  …
  -/
  rw [mem_toAddSubgroup, mem_graph_iff, Eq.comm]
  /-
    case h.e'_3.h.h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    hL₁ : Function.Surjective (Function.comp Prod.fst ⇑L.subtype)
    hL₂ : Function.Surjective (Function.comp Prod.snd ⇑L.subtype)
    e : AddEquiv (HasQuotient.Quotient M L.goursatFst) (HasQuotient.Quotient N L.g …
    he : Eq (((QuotientAddGroup.mk' L.toAddSubgroup.goursatFst).prodMap (QuotientA …
    this : ∀ (r : R) (x : HasQuotient.Quotient M L.goursatFst), Eq (e (HSMul.hSMul …
    e_1✝ : Eq (AddSubgroup (Prod (HasQuotient.Quotient M L.goursatFst) (HasQuotien …
    v : Prod (HasQuotient.Quotient M L.goursatFst) (HasQuotient.Quotient N L.gours …
    ⊢ Iff (Eq (↑{ toFun := e.toFun, map_add' := ⋯, map_smul' := this, invFun := e. …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- **Goursat's lemma** for an arbitrary submodule of a product.

If `L` is a submodule of `M × N`, then there exist submodules `M'' ≤ M' ≤ M` and `N'' ≤ N' ≤ N` such
that `L ≤ M' × N'`, and `L` is (the image in `M × N` of) the preimage of the graph of an `R`-linear
isomorphism `M' ⧸ M'' ≃ N' ⧸ N''`. -/
lemma goursat : ∃ (M' : Submodule R M) (N' : Submodule R N) (M'' : Submodule R M')
    (N'' : Submodule R N') (e : (M' ⧸ M'') ≃ₗ[R] N' ⧸ N''),
    L = (e.graph.comap <| M''.mkQ.prodMap N''.mkQ).map (M'.subtype.prodMap N'.subtype) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    ⊢ Exists fun M' => Exists fun N' => Exists fun M'' => Exists fun N'' => Exists …
  -/
  let M' := L.map (LinearMap.fst ..)
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
    ⊢ Exists fun M' => Exists fun N' => Exists fun M'' => Exists fun N'' => Exists …
  -/
  let N' := L.map (LinearMap.snd ..)
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
    N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
    ⊢ Exists fun M' => Exists fun N' => Exists fun M'' => Exists fun N'' => Exists …
  -/
  let P : L →ₗ[R] M' := (LinearMap.fst ..).submoduleMap L
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
    N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
    P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    ⊢ Exists fun M' => Exists fun N' => Exists fun M'' => Exists fun N'' => Exists …
  -/
  let Q : L →ₗ[R] N' := (LinearMap.snd ..).submoduleMap L
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
    N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
    P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    ⊢ Exists fun M' => Exists fun N' => Exists fun M'' => Exists fun N'' => Exists …
  -/
  let L' : Submodule R (M' × N') := LinearMap.range (P.prod Q)
  have hL₁' : Surjective (Prod.fst ∘ L'.subtype) := by
    simp only [← coe_fst (R := R), ← coe_comp, ← range_eq_top, LinearMap.range_comp, range_subtype]
    simpa only [L', ← LinearMap.range_comp, fst_prod, range_eq_top] using
      (LinearMap.fst ..).submoduleMap_surjective L
  have hL₂' : Surjective (Prod.snd ∘ L'.subtype) := by
    simp only [← coe_snd (R := R), ← coe_comp, ← range_eq_top, LinearMap.range_comp, range_subtype]
    simpa only [L', ← LinearMap.range_comp, snd_prod, range_eq_top] using
      (LinearMap.snd ..).submoduleMap_surjective L
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
    N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
    P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
    hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
    hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
    ⊢ Exists fun M' => Exists fun N' => Exists fun M'' => Exists fun N'' => Exists …
  -/
  obtain ⟨e, he⟩ := goursat_surjective hL₁' hL₂'
  /-
    case intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
    N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
    P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
    hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
    hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
    e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
    he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
    ⊢ Exists fun M' => Exists fun N' => Exists fun M'' => Exists fun N'' => Exists …
  -/
  use M', N', L'.goursatFst, L'.goursatSnd, e
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
    N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
    P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
    hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
    hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
    e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
    he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
    ⊢ Eq L (Submodule.map (M'.subtype.prodMap N'.subtype) (Submodule.comap (L'.gou …
  -/
  rw [← he]
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
    N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
    P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
    hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
    hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
    e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
    he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
    ⊢ Eq L (Submodule.map (M'.subtype.prodMap N'.subtype) (Submodule.comap (L'.gou …
  -/
  simp only [LinearMap.range_comp, Submodule.range_subtype, L']
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    L : Submodule R (Prod M N)
    M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
    N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
    P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
    L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
    hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
    hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
    e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
    he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
    ⊢ Eq L (Submodule.map (M'.subtype.prodMap N'.subtype) (Submodule.comap ((Linea …
  -/
  rw [comap_map_eq_self]
    /-
      case h
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      L : Submodule R (Prod M N)
      M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
      N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
      P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
      Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
      L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
      hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
      hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
      e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
      he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
      ⊢ Eq L (Submodule.map (M'.subtype.prodMap N'.subtype) (LinearMap.range (P.prod …
    -/
  · ext ⟨m, n⟩
    /-
      case h.h.mk
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      L : Submodule R (Prod M N)
      M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
      N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
      P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
      Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
      L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
      hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
      hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
      e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
      he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
      m : M
      n : N
      ⊢ Iff (Membership.mem L { fst := m, snd := n }) (Membership.mem (Submodule.map …
    -/
    constructor
      /-
        case h.h.mk.mp
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        L : Submodule R (Prod M N)
        M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
        N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
        P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
        Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
        L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
        hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
        hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
        e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
        he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
        m : M
        n : N
        ⊢ Membership.mem L { fst := m, snd := n } → Membership.mem (Submodule.map (M'. …
      -/
    · intro hmn
      simp only [mem_map, LinearMap.mem_range, prod_apply, Subtype.exists, Prod.exists, coe_prodMap,
        coe_subtype, Prod.map_apply, Prod.mk.injEq, exists_and_right, exists_eq_right_right,
        exists_eq_right, M', N', fst_apply, snd_apply]
      /-
        case h.h.mk.mp
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        L : Submodule R (Prod M N)
        M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
        N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
        P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
        Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
        L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
        hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
        hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
        e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
        he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
        m : M
        n : N
        hmn : Membership.mem L { fst := m, snd := n }
        ⊢ Exists fun x => Exists fun x_1 => Exists fun a => Exists fun b => Exists fun …
      -/
      exact ⟨⟨n, hmn⟩, ⟨m, hmn⟩, ⟨m, n, hmn, rfl⟩⟩
      /-
        🎉 no goals
      -/
    · simp only [mem_map, LinearMap.mem_range, prod_apply, Subtype.exists, Prod.exists,
        coe_prodMap, coe_subtype, Prod.map_apply, Prod.mk.injEq, exists_and_right,
        exists_eq_right_right, exists_eq_right, forall_exists_index, Pi.prod]
      /-
        case h.h.mk.mpr
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        L : Submodule R (Prod M N)
        M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
        N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
        P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
        Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
        L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
        hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
        hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
        e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
        he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
        m : M
        n : N
        ⊢ ∀ (x : Membership.mem M' m) (x_1 : Membership.mem N' n) (x_2 : M) (x_3 : N)  …
      -/
      rintro hm hn m₁ n₁ hm₁n₁ ⟨hP, hQ⟩
      /-
        case h.h.mk.mpr.intro
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        L : Submodule R (Prod M N)
        M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
        N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
        P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
        Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
        L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
        hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
        hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
        e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
        he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
        m : M
        n : N
        hm : Membership.mem M' m
        hn : Membership.mem N' n
        m₁ : M
        n₁ : N
        hm₁n₁ : Membership.mem L { fst := m₁, snd := n₁ }
        hP : Eq (P ⟨{ fst := m₁, snd := n₁ }, ⋯⟩) ⟨m, ⋯⟩
        hQ : Eq (Q ⟨{ fst := m₁, snd := n₁ }, ⋯⟩) ⟨n, ⋯⟩
        ⊢ Membership.mem L { fst := m, snd := n }
      -/
      simp only [Subtype.ext_iff] at hP hQ
      /-
        case h.h.mk.mpr.intro
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        L : Submodule R (Prod M N)
        M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
        N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
        P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
        Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
        L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
        hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
        hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
        e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
        he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
        m : M
        n : N
        hm : Membership.mem M' m
        hn : Membership.mem N' n
        m₁ : M
        n₁ : N
        hm₁n₁ : Membership.mem L { fst := m₁, snd := n₁ }
        hP : Eq (↑(P ⟨{ fst := m₁, snd := n₁ }, ⋯⟩)) m
        hQ : Eq (↑(Q ⟨{ fst := m₁, snd := n₁ }, ⋯⟩)) n
        ⊢ Membership.mem L { fst := m, snd := n }
      -/
      rwa [← hP, ← hQ]
      /-
        🎉 no goals
      -/
    /-
      case h
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      L : Submodule R (Prod M N)
      M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
      N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
      P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
      Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
      L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
      hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
      hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
      e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
      he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
      ⊢ LE.le (LinearMap.ker ((LinearMap.range (P.prod Q)).goursatFst.mkQ.prodMap (L …
    -/
  · convert goursatFst_prod_goursatSnd_le (range <| P.prod Q)
    /-
      case h.e'_3
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      L : Submodule R (Prod M N)
      M' : Submodule R M := Submodule.map (LinearMap.fst R M N) L
      N' : Submodule R N := Submodule.map (LinearMap.snd R M N) L
      P : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
      Q : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem L x) (Subtype fu …
      L' : Submodule R (Prod (Subtype fun x => Membership.mem M' x) (Subtype fun x = …
      hL₁' : Function.Surjective (Function.comp Prod.fst ⇑L'.subtype)
      hL₂' : Function.Surjective (Function.comp Prod.snd ⇑L'.subtype)
      e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Members …
      he : Eq (LinearMap.range ((L'.goursatFst.mkQ.prodMap L'.goursatSnd.mkQ).comp L …
      ⊢ Eq (LinearMap.ker ((LinearMap.range (P.prod Q)).goursatFst.mkQ.prodMap (Line …
    -/
    ext ⟨m, n⟩
    simp_rw [mem_ker, coe_prodMap, Prod.map_apply, Submodule.mem_prod, Prod.zero_eq_mk,
      Prod.ext_iff, ← mem_ker, ker_mkQ]


