/-- A partial `L`-equivalence, implemented as an equivalence between substructures. -/
structure PartialEquiv where
  /-- The substructure which is the domain of the equivalence. -/
  dom : L.Substructure M
  /-- The substructure which is the codomain of the equivalence. -/
  cod : L.Substructure N
  /-- The equivalence between the two subdomains. -/
  toEquiv : dom ≃[L] cod


@[inherit_doc]
scoped[FirstOrder] notation:25 M " ≃ₚ[" L "] " N =>
  FirstOrder.Language.PartialEquiv L M N


noncomputable instance instInhabited_self : Inhabited (M ≃ₚ[L] M) :=
  ⟨⊤, ⊤, Equiv.refl L (⊤ : L.Substructure M)⟩


/-- Maps to the symmetric partial equivalence. -/
def symm (f : M ≃ₚ[L] N) : N ≃ₚ[L] M where
  dom := f.cod
  cod := f.dom
  toEquiv := f.toEquiv.symm


@[simp]
theorem symm_symm (f : M ≃ₚ[L] N) : f.symm.symm = f :=
  rfl


@[simp]
theorem symm_apply (f : M ≃ₚ[L] N) (x : f.cod) : f.symm.toEquiv x = f.toEquiv.symm x :=
  rfl


instance : LE (M ≃ₚ[L] N) :=
  ⟨fun f g ↦ ∃ h : f.dom ≤ g.dom,
    (subtype _).comp (g.toEquiv.toEmbedding.comp (Substructure.inclusion h)) =
      (subtype _).comp f.toEquiv.toEmbedding⟩


theorem le_def (f g : M ≃ₚ[L] N) : f ≤ g ↔ ∃ h : f.dom ≤ g.dom,
    (subtype _).comp (g.toEquiv.toEmbedding.comp (Substructure.inclusion h)) =
      (subtype _).comp f.toEquiv.toEmbedding :=
  Iff.rfl


@[gcongr] theorem dom_le_dom {f g : M ≃ₚ[L] N} : f ≤ g → f.dom ≤ g.dom := fun ⟨le, _⟩ ↦ le


@[gcongr] theorem cod_le_cod {f g : M ≃ₚ[L] N} : f ≤ g → f.cod ≤ g.cod := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    ⊢ LE.le f g → LE.le f.cod g.cod
  -/
  rintro ⟨_, eq_fun⟩ n hn
  /-
    case intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    w✝ : LE.le f.dom g.dom
    eq_fun : Eq (g.cod.subtype.comp (g.toEquiv.toEmbedding.comp (FirstOrder.Langua …
    n : N
    hn : Membership.mem f.cod n
    ⊢ Membership.mem g.cod n
  -/
  let m := f.toEquiv.symm ⟨n, hn⟩
  have  : ((subtype _).comp f.toEquiv.toEmbedding) m = n := by simp only [m, Embedding.comp_apply,
    Equiv.coe_toEmbedding, Equiv.apply_symm_apply, coeSubtype]
  /-
    case intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    w✝ : LE.le f.dom g.dom
    eq_fun : Eq (g.cod.subtype.comp (g.toEquiv.toEmbedding.comp (FirstOrder.Langua …
    n : N
    hn : Membership.mem f.cod n
    m : Subtype fun x => Membership.mem f.dom x := f.toEquiv.symm ⟨n, hn⟩
    this : Eq ((f.cod.subtype.comp f.toEquiv.toEmbedding) m) n
    ⊢ Membership.mem g.cod n
  -/
  rw [← this, ← eq_fun]
  simp only [Embedding.comp_apply, coe_inclusion, Equiv.coe_toEmbedding, coeSubtype,
    SetLike.coe_mem]


theorem subtype_toEquiv_inclusion {f g : M ≃ₚ[L] N} (h : f ≤ g) :
    (subtype _).comp (g.toEquiv.toEmbedding.comp (Substructure.inclusion (dom_le_dom h))) =
      (subtype _).comp f.toEquiv.toEmbedding := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    h : LE.le f g
    ⊢ Eq (g.cod.subtype.comp (g.toEquiv.toEmbedding.comp (FirstOrder.Language.Subs …
  -/
  let ⟨_, eq⟩ := h; exact eq
                    /-
                      🎉 no goals
                    -/


theorem toEquiv_inclusion {f g : M ≃ₚ[L] N} (h : f ≤ g) :
    g.toEquiv.toEmbedding.comp (Substructure.inclusion (dom_le_dom h)) =
      (Substructure.inclusion (cod_le_cod h)).comp f.toEquiv.toEmbedding := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    h : LE.le f g
    ⊢ Eq (g.toEquiv.toEmbedding.comp (FirstOrder.Language.Substructure.inclusion ⋯ …
  -/
  rw [← (subtype _).comp_inj, subtype_toEquiv_inclusion h]
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    h : LE.le f g
    ⊢ Eq (f.cod.subtype.comp f.toEquiv.toEmbedding) (g.cod.subtype.comp ((FirstOrd …
  -/
  ext
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    h : LE.le f g
    x✝ : Subtype fun x => Membership.mem f.dom x
    ⊢ Eq ((f.cod.subtype.comp f.toEquiv.toEmbedding) x✝) ((g.cod.subtype.comp ((Fi …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem toEquiv_inclusion_apply {f g : M ≃ₚ[L] N} (h : f ≤ g) (x : f.dom) :
    g.toEquiv (Substructure.inclusion (dom_le_dom h) x) =
      Substructure.inclusion (cod_le_cod h) (f.toEquiv x) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    h : LE.le f g
    x : Subtype fun x => Membership.mem f.dom x
    ⊢ Eq (g.toEquiv ((FirstOrder.Language.Substructure.inclusion ⋯) x)) ((FirstOrd …
  -/
  apply (subtype _).injective
  /-
    case a
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    h : LE.le f g
    x : Subtype fun x => Membership.mem f.dom x
    ⊢ Eq (g.cod.subtype (g.toEquiv ((FirstOrder.Language.Substructure.inclusion ⋯) …
  -/
  change (subtype _).comp (g.toEquiv.toEmbedding.comp (inclusion _)) x = _
  /-
    case a
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    h : LE.le f g
    x : Subtype fun x => Membership.mem f.dom x
    ⊢ Eq ((g.cod.subtype.comp (g.toEquiv.toEmbedding.comp (FirstOrder.Language.Sub …
  -/
  rw [subtype_toEquiv_inclusion h]
  /-
    case a
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    h : LE.le f g
    x : Subtype fun x => Membership.mem f.dom x
    ⊢ Eq ((f.cod.subtype.comp f.toEquiv.toEmbedding) x) (g.cod.subtype ((FirstOrde …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem le_iff {f g : M ≃ₚ[L] N} : f ≤ g ↔
    ∃ dom_le_dom : f.dom ≤ g.dom,
    ∃ cod_le_cod : f.cod ≤ g.cod,
    ∀ x, inclusion cod_le_cod (f.toEquiv x) = g.toEquiv (inclusion dom_le_dom x) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    ⊢ Iff (LE.le f g) (Exists fun dom_le_dom => Exists fun cod_le_cod => ∀ (x : Su …
  -/
  constructor
  · exact fun h ↦ ⟨dom_le_dom h, cod_le_cod h,
      by intro x; apply (subtype _).inj'; rwa [toEquiv_inclusion_apply]⟩
    /-
      case mpr
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f g : L.PartialEquiv M N
      ⊢ (Exists fun dom_le_dom => Exists fun cod_le_cod => ∀ (x : Subtype fun x => M …
    -/
  · rintro ⟨dom_le_dom, le_cod, h_eq⟩
    /-
      case mpr.intro.intro
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f g : L.PartialEquiv M N
      dom_le_dom : LE.le f.dom g.dom
      le_cod : LE.le f.cod g.cod
      h_eq : ∀ (x : Subtype fun x => Membership.mem f.dom x), Eq ((FirstOrder.Langua …
      ⊢ LE.le f g
    -/
    rw [le_def]
    /-
      case mpr.intro.intro
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f g : L.PartialEquiv M N
      dom_le_dom : LE.le f.dom g.dom
      le_cod : LE.le f.cod g.cod
      h_eq : ∀ (x : Subtype fun x => Membership.mem f.dom x), Eq ((FirstOrder.Langua …
      ⊢ Exists fun h => Eq (g.cod.subtype.comp (g.toEquiv.toEmbedding.comp (FirstOrd …
    -/
    exact ⟨dom_le_dom, by ext; change subtype _ (g.toEquiv _) = _; rw [← h_eq]; rfl⟩
    /-
      🎉 no goals
    -/


theorem le_trans (f g h : M ≃ₚ[L] N) : f ≤ g → g ≤ h → f ≤ h := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g h : L.PartialEquiv M N
    ⊢ LE.le f g → LE.le g h → LE.le f h
  -/
  rintro ⟨le_fg, eq_fg⟩ ⟨le_gh, eq_gh⟩
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g h : L.PartialEquiv M N
    le_fg : LE.le f.dom g.dom
    eq_fg : Eq (g.cod.subtype.comp (g.toEquiv.toEmbedding.comp (FirstOrder.Languag …
    le_gh : LE.le g.dom h.dom
    eq_gh : Eq (h.cod.subtype.comp (h.toEquiv.toEmbedding.comp (FirstOrder.Languag …
    ⊢ LE.le f h
  -/
  refine ⟨le_fg.trans le_gh, ?_⟩
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g h : L.PartialEquiv M N
    le_fg : LE.le f.dom g.dom
    eq_fg : Eq (g.cod.subtype.comp (g.toEquiv.toEmbedding.comp (FirstOrder.Languag …
    le_gh : LE.le g.dom h.dom
    eq_gh : Eq (h.cod.subtype.comp (h.toEquiv.toEmbedding.comp (FirstOrder.Languag …
    ⊢ Eq (h.cod.subtype.comp (h.toEquiv.toEmbedding.comp (FirstOrder.Language.Subs …
  -/
  rw [← eq_fg, ← Embedding.comp_assoc (g := g.toEquiv.toEmbedding), ← eq_gh]
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g h : L.PartialEquiv M N
    le_fg : LE.le f.dom g.dom
    eq_fg : Eq (g.cod.subtype.comp (g.toEquiv.toEmbedding.comp (FirstOrder.Languag …
    le_gh : LE.le g.dom h.dom
    eq_gh : Eq (h.cod.subtype.comp (h.toEquiv.toEmbedding.comp (FirstOrder.Languag …
    ⊢ Eq (h.cod.subtype.comp (h.toEquiv.toEmbedding.comp (FirstOrder.Language.Subs …
  -/
  ext
  /-
    case intro.intro.h
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g h : L.PartialEquiv M N
    le_fg : LE.le f.dom g.dom
    eq_fg : Eq (g.cod.subtype.comp (g.toEquiv.toEmbedding.comp (FirstOrder.Languag …
    le_gh : LE.le g.dom h.dom
    eq_gh : Eq (h.cod.subtype.comp (h.toEquiv.toEmbedding.comp (FirstOrder.Languag …
    x✝ : Subtype fun x => Membership.mem f.dom x
    ⊢ Eq ((h.cod.subtype.comp (h.toEquiv.toEmbedding.comp (FirstOrder.Language.Sub …
  -/
  simp
  /-
    🎉 no goals
  -/


private theorem le_refl (f : M ≃ₚ[L] N) : f ≤ f := ⟨le_rfl, rfl⟩


private theorem le_antisymm (f g : M ≃ₚ[L] N) (le_fg : f ≤ g) (le_gf : g ≤ f) : f = g := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    le_fg : LE.le f g
    le_gf : LE.le g f
    ⊢ Eq f g
  -/
  let ⟨dom_f, cod_f, equiv_f⟩ := f
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    dom_f : L.Substructure M
    cod_f : L.Substructure N
    equiv_f : L.Equiv (Subtype fun x => Membership.mem dom_f x) (Subtype fun x =>  …
    le_fg : LE.le { dom := dom_f, cod := cod_f, toEquiv := equiv_f } g
    le_gf : LE.le g { dom := dom_f, cod := cod_f, toEquiv := equiv_f }
    ⊢ Eq { dom := dom_f, cod := cod_f, toEquiv := equiv_f } g
  -/
  cases _root_.le_antisymm (dom_le_dom le_fg) (dom_le_dom le_gf)
  /-
    case refl
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    cod_f : L.Substructure N
    equiv_f : L.Equiv (Subtype fun x => Membership.mem g.1 x) (Subtype fun x => Me …
    le_fg : LE.le { dom := g.1, cod := cod_f, toEquiv := equiv_f } g
    le_gf : LE.le g { dom := g.1, cod := cod_f, toEquiv := equiv_f }
    ⊢ Eq { dom := g.1, cod := cod_f, toEquiv := equiv_f } g
  -/
  cases _root_.le_antisymm (cod_le_cod le_fg) (cod_le_cod le_gf)
  /-
    case refl.refl
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    equiv_f : L.Equiv (Subtype fun x => Membership.mem g.1 x) (Subtype fun x => Me …
    le_fg : LE.le { dom := g.1, cod := g.2, toEquiv := equiv_f } g
    le_gf : LE.le g { dom := g.1, cod := g.2, toEquiv := equiv_f }
    ⊢ Eq { dom := g.1, cod := g.2, toEquiv := equiv_f } g
  -/
  convert rfl
  /-
    case h.e'_3.h.e'_8.h.h.h.h.h
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    equiv_f : L.Equiv (Subtype fun x => Membership.mem g.1 x) (Subtype fun x => Me …
    le_fg : LE.le { dom := g.1, cod := g.2, toEquiv := equiv_f } g
    le_gf : LE.le g { dom := g.1, cod := g.2, toEquiv := equiv_f }
    he✝¹ : Eq g.dom g.1
    he✝ : Eq g.cod g.2
    ⊢ Eq g.toEquiv equiv_f
  -/
  exact Equiv.injective_toEmbedding ((subtype _).comp_injective (subtype_toEquiv_inclusion le_fg))
  /-
    🎉 no goals
  -/


instance : PartialOrder (M ≃ₚ[L] N) where
  le_refl := le_refl
  le_trans := le_trans
  le_antisymm := le_antisymm


@[gcongr] lemma symm_le_symm {f g : M ≃ₚ[L] N} (hfg : f ≤ g) : f.symm ≤ g.symm := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    hfg : LE.le f g
    ⊢ LE.le f.symm g.symm
  -/
  rw [le_iff]
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    hfg : LE.le f g
    ⊢ Exists fun dom_le_dom => Exists fun cod_le_cod => ∀ (x : Subtype fun x => Me …
  -/
  refine ⟨cod_le_cod hfg, dom_le_dom hfg, ?_⟩
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    hfg : LE.le f g
    ⊢ ∀ (x : Subtype fun x => Membership.mem f.symm.dom x), Eq ((FirstOrder.Langua …
  -/
  intro x
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    hfg : LE.le f g
    x : Subtype fun x => Membership.mem f.symm.dom x
    ⊢ Eq ((FirstOrder.Language.Substructure.inclusion ⋯) (f.symm.toEquiv x)) (g.sy …
  -/
  apply g.toEquiv.injective
  /-
    case a
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    hfg : LE.le f g
    x : Subtype fun x => Membership.mem f.symm.dom x
    ⊢ Eq (g.toEquiv ((FirstOrder.Language.Substructure.inclusion ⋯) (f.symm.toEqui …
  -/
  change g.toEquiv (inclusion _ (f.toEquiv.symm x)) = g.toEquiv (g.toEquiv.symm _)
  rw [g.toEquiv.apply_symm_apply, (Equiv.apply_symm_apply f.toEquiv x).symm,
    f.toEquiv.symm_apply_apply]
  /-
    case a
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    hfg : LE.le f g
    x : Subtype fun x => Membership.mem f.symm.dom x
    ⊢ Eq (g.toEquiv ((FirstOrder.Language.Substructure.inclusion ⋯) (f.toEquiv.sym …
  -/
  exact toEquiv_inclusion_apply hfg _
  /-
    🎉 no goals
  -/


theorem monotone_symm : Monotone (fun (f : M ≃ₚ[L] N) ↦ f.symm) := fun _ _ => symm_le_symm


theorem symm_le_iff {f : M ≃ₚ[L] N} {g : N ≃ₚ[L] M} : f.symm ≤ g ↔ f ≤ g.symm :=
      /-
        L : FirstOrder.Language
        M : Type w
        N : Type w'
        inst✝¹ : L.Structure M
        inst✝ : L.Structure N
        f : L.PartialEquiv M N
        g : L.PartialEquiv N M
        ⊢ LE.le f.symm g → LE.le f g.symm
      -/
  ⟨by intro h; rw [← f.symm_symm]; exact monotone_symm h,
                                   /-
                                     🎉 no goals
                                   -/
       /-
         L : FirstOrder.Language
         M : Type w
         N : Type w'
         inst✝¹ : L.Structure M
         inst✝ : L.Structure N
         f : L.PartialEquiv M N
         g : L.PartialEquiv N M
         ⊢ LE.le f g.symm → LE.le f.symm g
       -/
    by intro h; rw  [← g.symm_symm]; exact monotone_symm h⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem ext {f g : M ≃ₚ[L] N} (h_dom : f.dom = g.dom) : (∀ x : M, ∀ h : x ∈ f.dom,
    subtype _ (f.toEquiv ⟨x, h⟩) = subtype _ (g.toEquiv ⟨x, (h_dom ▸ h)⟩)) → f = g := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    h_dom : Eq f.dom g.dom
    ⊢ (∀ (x : M) (h : Membership.mem f.dom x), Eq (f.cod.subtype (f.toEquiv ⟨x, h⟩ …
  -/
  intro h
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    h_dom : Eq f.dom g.dom
    h : ∀ (x : M) (h : Membership.mem f.dom x), Eq (f.cod.subtype (f.toEquiv ⟨x, h …
    ⊢ Eq f g
  -/
  rcases f with ⟨dom_f, cod_f, equiv_f⟩
  /-
    case mk
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.PartialEquiv M N
    dom_f : L.Substructure M
    cod_f : L.Substructure N
    equiv_f : L.Equiv (Subtype fun x => Membership.mem dom_f x) (Subtype fun x =>  …
    h_dom : Eq { dom := dom_f, cod := cod_f, toEquiv := equiv_f }.dom g.dom
    h : ∀ (x : M) (h : Membership.mem { dom := dom_f, cod := cod_f, toEquiv := equ …
    ⊢ Eq { dom := dom_f, cod := cod_f, toEquiv := equiv_f } g
  -/
  cases h_dom
  /-
    case mk.refl
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.PartialEquiv M N
    cod_f : L.Substructure N
    equiv_f : L.Equiv (Subtype fun x => Membership.mem g.1 x) (Subtype fun x => Me …
    h : ∀ (x : M) (h : Membership.mem { dom := g.1, cod := cod_f, toEquiv := equiv …
    ⊢ Eq { dom := g.1, cod := cod_f, toEquiv := equiv_f } g
  -/
  apply le_antisymm <;> (rw [le_def]; use le_rfl; ext ⟨x, hx⟩)
    /-
      case h.h.mk
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      g : L.PartialEquiv M N
      cod_f : L.Substructure N
      equiv_f : L.Equiv (Subtype fun x => Membership.mem g.1 x) (Subtype fun x => Me …
      h : ∀ (x : M) (h : Membership.mem { dom := g.1, cod := cod_f, toEquiv := equiv …
      x : M
      hx : Membership.mem { dom := g.1, cod := cod_f, toEquiv := equiv_f }.dom x
      ⊢ Eq ((g.cod.subtype.comp (g.toEquiv.toEmbedding.comp (FirstOrder.Language.Sub …
    -/
  · exact (h x hx).symm
    /-
      🎉 no goals
    -/
    /-
      case h.h.mk
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      g : L.PartialEquiv M N
      cod_f : L.Substructure N
      equiv_f : L.Equiv (Subtype fun x => Membership.mem g.1 x) (Subtype fun x => Me …
      h : ∀ (x : M) (h : Membership.mem { dom := g.1, cod := cod_f, toEquiv := equiv …
      x : M
      hx : Membership.mem g.dom x
      ⊢ Eq (({ dom := g.1, cod := cod_f, toEquiv := equiv_f }.cod.subtype.comp ({ do …
    -/
  · exact h x hx
    /-
      🎉 no goals
    -/


theorem ext_iff {f g : M ≃ₚ[L] N} : f = g ↔ ∃ h_dom : f.dom = g.dom,
    ∀ x : M, ∀ h : x ∈ f.dom,
    subtype _ (f.toEquiv ⟨x, h⟩) = subtype _ (g.toEquiv ⟨x, (h_dom ▸ h)⟩) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f g : L.PartialEquiv M N
    ⊢ Iff (Eq f g) (Exists fun h_dom => ∀ (x : M) (h : Membership.mem f.dom x), Eq …
  -/
  constructor
    /-
      case mp
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f g : L.PartialEquiv M N
      ⊢ Eq f g → Exists fun h_dom => ∀ (x : M) (h : Membership.mem f.dom x), Eq (f.c …
    -/
  · intro h_eq
    /-
      case mp
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f g : L.PartialEquiv M N
      h_eq : Eq f g
      ⊢ Exists fun h_dom => ∀ (x : M) (h : Membership.mem f.dom x), Eq (f.cod.subtyp …
    -/
    rcases f with ⟨dom_f, cod_f, equiv_f⟩
    /-
      case mp.mk
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      g : L.PartialEquiv M N
      dom_f : L.Substructure M
      cod_f : L.Substructure N
      equiv_f : L.Equiv (Subtype fun x => Membership.mem dom_f x) (Subtype fun x =>  …
      h_eq : Eq { dom := dom_f, cod := cod_f, toEquiv := equiv_f } g
      ⊢ Exists fun h_dom => ∀ (x : M) (h : Membership.mem { dom := dom_f, cod := cod …
    -/
    cases h_eq
    /-
      case mp.mk.refl
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      dom_f : L.Substructure M
      cod_f : L.Substructure N
      equiv_f : L.Equiv (Subtype fun x => Membership.mem dom_f x) (Subtype fun x =>  …
      ⊢ Exists fun h_dom => ∀ (x : M) (h : Membership.mem { dom := dom_f, cod := cod …
    -/
    exact ⟨rfl, fun _ _ ↦ rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      f g : L.PartialEquiv M N
      ⊢ (Exists fun h_dom => ∀ (x : M) (h : Membership.mem f.dom x), Eq (f.cod.subty …
    -/
  · rintro ⟨h, H⟩; exact ext h H
                   /-
                     🎉 no goals
                   -/


theorem monotone_dom : Monotone (fun f : M ≃ₚ[L] N ↦ f.dom) := fun _ _ ↦ dom_le_dom


theorem monotone_cod : Monotone (fun f : M ≃ₚ[L] N ↦ f.cod) := fun _ _ ↦ cod_le_cod


/-- Restriction of a partial equivalence to a substructure of the domain. -/
noncomputable def domRestrict (f : M ≃ₚ[L] N) {A : L.Substructure M} (h : A ≤ f.dom) :
    M ≃ₚ[L] N := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.PartialEquiv M N
    A : L.Substructure M
    h : LE.le A f.dom
    ⊢ L.PartialEquiv M N
  -/
  let g := (subtype _).comp (f.toEquiv.toEmbedding.comp (A.inclusion h))
  exact {
    dom := A
    cod := g.toHom.range
    toEquiv := g.equivRange
  }


theorem domRestrict_le (f : M ≃ₚ[L] N) {A : L.Substructure M} (h : A ≤ f.dom) :
    f.domRestrict h ≤ f := ⟨h, rfl⟩


theorem le_domRestrict (f g : M ≃ₚ[L] N) {A : L.Substructure M} (hf : f.dom ≤ A)
    (hg : A ≤ g.dom) (hfg : f ≤ g) : f ≤ g.domRestrict hg :=
          /-
            L : FirstOrder.Language
            M : Type w
            N : Type w'
            inst✝¹ : L.Structure M
            inst✝ : L.Structure N
            f g : L.PartialEquiv M N
            A : L.Substructure M
            hf : LE.le f.dom A
            hg : LE.le A g.dom
            hfg : LE.le f g
            ⊢ Eq ((g.domRestrict hg).cod.subtype.comp ((g.domRestrict hg).toEquiv.toEmbedd …
          -/
  ⟨hf, by rw [← (subtype_toEquiv_inclusion hfg)]; rfl⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- Restriction of a partial equivalence to a substructure of the codomain. -/
noncomputable def codRestrict (f : M ≃ₚ[L] N) {A : L.Substructure N} (h : A ≤ f.cod) :
    M ≃ₚ[L] N :=
  (f.symm.domRestrict h).symm


theorem codRestrict_le (f : M ≃ₚ[L] N) {A : L.Substructure N} (h : A ≤ f.cod) :
    codRestrict f h ≤ f :=
  symm_le_iff.2 (f.symm.domRestrict_le h)


theorem le_codRestrict (f g : M ≃ₚ[L] N) {A : L.Substructure N} (hf : f.cod ≤ A)
    (hg : A ≤ g.cod) (hfg : f ≤ g) : f ≤ g.codRestrict hg :=
  symm_le_iff.1 (le_domRestrict f.symm g.symm hf hg (monotone_symm hfg))


/-- A partial equivalence as an embedding from its domain. -/
def toEmbedding (f : M ≃ₚ[L] N) : f.dom ↪[L] N :=
  (subtype _).comp f.toEquiv.toEmbedding


@[simp]
theorem toEmbedding_apply {f : M ≃ₚ[L] N} (m : f.dom) :
    f.toEmbedding m = f.toEquiv m :=
  rfl


/-- Given a partial equivalence which has the whole structure as domain,
  returns the corresponding embedding. -/
def toEmbeddingOfEqTop {f : M ≃ₚ[L] N} (h : f.dom = ⊤) : M ↪[L] N :=
  (h ▸ f.toEmbedding).comp topEquiv.symm.toEmbedding


@[simp]
theorem toEmbeddingOfEqTop__apply {f : M ≃ₚ[L] N} (h : f.dom = ⊤) (m : M) :
    toEmbeddingOfEqTop h m = f.toEquiv ⟨m, h.symm ▸ mem_top m⟩ := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.PartialEquiv M N
    h : Eq f.dom Top.top
    m : M
    ⊢ Eq ((FirstOrder.Language.PartialEquiv.toEmbeddingOfEqTop h) m) ↑(f.toEquiv ⟨ …
  -/
  rcases f with ⟨dom, cod, g⟩
  /-
    case mk
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    m : M
    dom : L.Substructure M
    cod : L.Substructure N
    g : L.Equiv (Subtype fun x => Membership.mem dom x) (Subtype fun x => Membersh …
    h : Eq { dom := dom, cod := cod, toEquiv := g }.dom Top.top
    ⊢ Eq ((FirstOrder.Language.PartialEquiv.toEmbeddingOfEqTop h) m) ↑({ dom := do …
  -/
  cases h
  /-
    case mk.refl
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    m : M
    cod : L.Substructure N
    g : L.Equiv (Subtype fun x => Membership.mem { carrier := Set.univ, fun_mem := …
    ⊢ Eq ((FirstOrder.Language.PartialEquiv.toEmbeddingOfEqTop ⋯) m) ↑({ dom := {  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a partial equivalence which has the whole structure as domain and
  as codomain, returns the corresponding equivalence. -/
def toEquivOfEqTop {f : M ≃ₚ[L] N} (h_dom : f.dom = ⊤)
    (h_cod : f.cod = ⊤) : M ≃[L] N :=
  (topEquiv (M := N)).comp ((h_dom ▸ h_cod ▸ f.toEquiv).comp (topEquiv (M := M)).symm)


@[simp]
theorem toEquivOfEqTop_toEmbedding {f : M ≃ₚ[L] N} (h_dom : f.dom = ⊤)
    (h_cod : f.cod = ⊤) :
    (toEquivOfEqTop h_dom h_cod).toEmbedding = toEmbeddingOfEqTop h_dom := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.PartialEquiv M N
    h_dom : Eq f.dom Top.top
    h_cod : Eq f.cod Top.top
    ⊢ Eq (FirstOrder.Language.PartialEquiv.toEquivOfEqTop h_dom h_cod).toEmbedding …
  -/
  rcases f with ⟨dom, cod, g⟩
  /-
    case mk
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    dom : L.Substructure M
    cod : L.Substructure N
    g : L.Equiv (Subtype fun x => Membership.mem dom x) (Subtype fun x => Membersh …
    h_dom : Eq { dom := dom, cod := cod, toEquiv := g }.dom Top.top
    h_cod : Eq { dom := dom, cod := cod, toEquiv := g }.cod Top.top
    ⊢ Eq (FirstOrder.Language.PartialEquiv.toEquivOfEqTop h_dom h_cod).toEmbedding …
  -/
  cases h_dom
  /-
    case mk.refl
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    cod : L.Substructure N
    g : L.Equiv (Subtype fun x => Membership.mem { carrier := Set.univ, fun_mem := …
    h_cod : Eq { dom := { carrier := Set.univ, fun_mem := ⋯ }, cod := cod, toEquiv …
    ⊢ Eq (FirstOrder.Language.PartialEquiv.toEquivOfEqTop ⋯ h_cod).toEmbedding (Fi …
  -/
  cases h_cod
  /-
    case mk.refl.refl
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.Equiv (Subtype fun x => Membership.mem { carrier := Set.univ, fun_mem := …
    ⊢ Eq (FirstOrder.Language.PartialEquiv.toEquivOfEqTop ⋯ ⋯).toEmbedding (FirstO …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem dom_fg_iff_cod_fg {N : Type*} [L.Structure N] (f : M ≃ₚ[L] N) :
    f.dom.FG ↔ f.cod.FG := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    N : Type u_1
    inst✝ : L.Structure N
    f : L.PartialEquiv M N
    ⊢ Iff f.dom.FG f.cod.FG
  -/
  rw [Substructure.fg_iff_structure_fg, f.toEquiv.fg_iff, Substructure.fg_iff_structure_fg]
  /-
    🎉 no goals
  -/


/-- Given an embedding, returns the corresponding partial equivalence with `⊤` as domain. -/
noncomputable def toPartialEquiv (f : M ↪[L] N) : M ≃ₚ[L] N :=
  ⟨⊤, f.toHom.range, f.equivRange.comp (Substructure.topEquiv)⟩


theorem toPartialEquiv_injective :
    Function.Injective (fun f : M ↪[L] N ↦ f.toPartialEquiv) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    ⊢ Function.Injective fun f => f.toPartialEquiv
  -/
  intro _ _ h
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    a₁✝ a₂✝ : L.Embedding M N
    h : Eq ((fun f => f.toPartialEquiv) a₁✝) ((fun f => f.toPartialEquiv) a₂✝)
    ⊢ Eq a₁✝ a₂✝
  -/
  ext
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    a₁✝ a₂✝ : L.Embedding M N
    h : Eq ((fun f => f.toPartialEquiv) a₁✝) ((fun f => f.toPartialEquiv) a₂✝)
    x✝ : M
    ⊢ Eq (a₁✝ x✝) (a₂✝ x✝)
  -/
  rw [PartialEquiv.ext_iff] at h
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    a₁✝ a₂✝ : L.Embedding M N
    h : Exists fun h_dom => ∀ (x : M) (h : Membership.mem ((fun f => f.toPartialEq …
    x✝ : M
    ⊢ Eq (a₁✝ x✝) (a₂✝ x✝)
  -/
  rcases h with ⟨_, H⟩
  /-
    case h.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    a₁✝ a₂✝ : L.Embedding M N
    x✝ : M
    w✝ : Eq ((fun f => f.toPartialEquiv) a₁✝).dom ((fun f => f.toPartialEquiv) a₂✝ …
    H : ∀ (x : M) (h : Membership.mem ((fun f => f.toPartialEquiv) a₁✝).dom x), Eq …
    ⊢ Eq (a₁✝ x✝) (a₂✝ x✝)
  -/
  exact H _ (Substructure.mem_top _)
  /-
    🎉 no goals
  -/


@[simp]
theorem toEmbedding_toPartialEquiv (f : M ↪[L] N) :
    PartialEquiv.toEmbeddingOfEqTop (f := f.toPartialEquiv) rfl = f :=
  rfl


@[simp]
theorem toPartialEquiv_toEmbedding {f :  M ≃ₚ[L] N} (h : f.dom = ⊤) :
    (PartialEquiv.toEmbeddingOfEqTop h).toPartialEquiv = f := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.PartialEquiv M N
    h : Eq f.dom Top.top
    ⊢ Eq (FirstOrder.Language.PartialEquiv.toEmbeddingOfEqTop h).toPartialEquiv f
  -/
  rcases f with ⟨_, _, _⟩
  /-
    case mk
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    dom✝ : L.Substructure M
    cod✝ : L.Substructure N
    toEquiv✝ : L.Equiv (Subtype fun x => Membership.mem dom✝ x) (Subtype fun x =>  …
    h : Eq { dom := dom✝, cod := cod✝, toEquiv := toEquiv✝ }.dom Top.top
    ⊢ Eq (FirstOrder.Language.PartialEquiv.toEmbeddingOfEqTop h).toPartialEquiv {  …
  -/
  cases h
  /-
    case mk.refl
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    cod✝ : L.Substructure N
    toEquiv✝ : L.Equiv (Subtype fun x => Membership.mem { carrier := Set.univ, fun …
    ⊢ Eq (FirstOrder.Language.PartialEquiv.toEmbeddingOfEqTop ⋯).toPartialEquiv {  …
  -/
  apply PartialEquiv.ext
    /-
      case mk.refl.a
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      cod✝ : L.Substructure N
      toEquiv✝ : L.Equiv (Subtype fun x => Membership.mem { carrier := Set.univ, fun …
      ⊢ ∀ (x : M) (h : Membership.mem (FirstOrder.Language.PartialEquiv.toEmbeddingO …
    -/
  · intro _ _
    /-
      case mk.refl.a
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      cod✝ : L.Substructure N
      toEquiv✝ : L.Equiv (Subtype fun x => Membership.mem { carrier := Set.univ, fun …
      x✝ : M
      h✝ : Membership.mem (FirstOrder.Language.PartialEquiv.toEmbeddingOfEqTop ⋯).to …
      ⊢ Eq ((FirstOrder.Language.PartialEquiv.toEmbeddingOfEqTop ⋯).toPartialEquiv.c …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mk.refl.h_dom
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      cod✝ : L.Substructure N
      toEquiv✝ : L.Equiv (Subtype fun x => Membership.mem { carrier := Set.univ, fun …
      ⊢ Eq (FirstOrder.Language.PartialEquiv.toEmbeddingOfEqTop ⋯).toPartialEquiv.do …
    -/
  · rfl
    /-
      🎉 no goals
    -/


instance : DirectedSystem (fun i ↦ (S i).dom)
    (fun _ _ h ↦ Substructure.inclusion (dom_le_dom (S.monotone h))) where
  map_self _ _ := rfl
  map_map _ _ _ _ _ _ := rfl


instance : DirectedSystem (fun i ↦ (S i).cod)
    (fun _ _ h ↦ Substructure.inclusion (cod_le_cod (S.monotone h))) where
  map_self _ _ := rfl
  map_map _ _ _ _ _ _ := rfl


/-- The limit of a directed system of PartialEquivs. -/
noncomputable def partialEquivLimit : M ≃ₚ[L] N where
  dom := iSup (fun i ↦ (S i).dom)
  cod := iSup (fun i ↦ (S i).cod)
  toEquiv :=
    (Equiv_iSup {
      toFun := (fun i ↦ (S i).cod)
      monotone' := monotone_cod.comp S.monotone}
    ).comp
      ((DirectLimit.equiv_lift L ι (fun i ↦ (S i).dom)
        (fun _ _ hij ↦ Substructure.inclusion (dom_le_dom (S.monotone hij)))
        (fun i ↦ (S i).cod)
        (fun _ _ hij ↦ Substructure.inclusion (cod_le_cod (S.monotone hij)))
        (fun i ↦ (S i).toEquiv)
        (fun _ _ hij _ ↦ toEquiv_inclusion_apply (S.monotone hij) _)
      ).comp
        (Equiv_iSup {
          toFun := (fun i ↦ (S i).dom)
          monotone' := monotone_dom.comp S.monotone}).symm)


@[simp]
theorem dom_partialEquivLimit : (partialEquivLimit S).dom = iSup (fun x ↦ (S x).dom) := rfl


@[simp]
theorem cod_partialEquivLimit : (partialEquivLimit S).cod = iSup (fun x ↦ (S x).cod) := rfl


@[simp]
lemma partialEquivLimit_comp_inclusion {i : ι} :
    (partialEquivLimit S).toEquiv.toEmbedding.comp (Substructure.inclusion (le_iSup _ i)) =
    (Substructure.inclusion (le_iSup _ i)).comp (S i).toEquiv.toEmbedding := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝⁴ : L.Structure M
    inst✝³ : L.Structure N
    ι : Type u_1
    inst✝² : Preorder ι
    inst✝¹ : Nonempty ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : OrderHom ι (L.PartialEquiv M N)
    i : ι
    ⊢ Eq ((FirstOrder.Language.DirectLimit.partialEquivLimit S).toEquiv.toEmbeddin …
  -/
  simp only [partialEquivLimit, Equiv.comp_toEmbedding, Embedding.comp_assoc]
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝⁴ : L.Structure M
    inst✝³ : L.Structure N
    ι : Type u_1
    inst✝² : Preorder ι
    inst✝¹ : Nonempty ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : OrderHom ι (L.PartialEquiv M N)
    i : ι
    ⊢ Eq ((FirstOrder.Language.DirectLimit.Equiv_iSup { toFun := fun i => (S i).co …
  -/
  rw [Equiv_isup_symm_inclusion]
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝⁴ : L.Structure M
    inst✝³ : L.Structure N
    ι : Type u_1
    inst✝² : Preorder ι
    inst✝¹ : Nonempty ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : OrderHom ι (L.PartialEquiv M N)
    i : ι
    ⊢ Eq ((FirstOrder.Language.DirectLimit.Equiv_iSup { toFun := fun i => (S i).co …
  -/
  congr
  /-
    🎉 no goals
  -/


theorem le_partialEquivLimit (i : ι) : S i ≤ partialEquivLimit S :=
  ⟨le_iSup (f := fun i ↦ (S i).dom) _, by
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝⁴ : L.Structure M
      inst✝³ : L.Structure N
      ι : Type u_1
      inst✝² : Preorder ι
      inst✝¹ : Nonempty ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      S : OrderHom ι (L.PartialEquiv M N)
      i : ι
      ⊢ Eq ((FirstOrder.Language.DirectLimit.partialEquivLimit S).cod.subtype.comp ( …
    -/
    #adaptation_note /-- After https://github.com/leanprover/lean4/pull/5020, these two `simp` calls cannot be combined. -/
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝⁴ : L.Structure M
      inst✝³ : L.Structure N
      ι : Type u_1
      inst✝² : Preorder ι
      inst✝¹ : Nonempty ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      S : OrderHom ι (L.PartialEquiv M N)
      i : ι
      ⊢ Eq ((FirstOrder.Language.DirectLimit.partialEquivLimit S).cod.subtype.comp ( …
    -/
    simp only [partialEquivLimit_comp_inclusion]
    simp only [cod_partialEquivLimit, dom_partialEquivLimit, ← Embedding.comp_assoc,
      subtype_comp_inclusion]⟩


/-- The type of equivalences between finitely generated substructures. -/
abbrev FGEquiv := {f : M ≃ₚ[L] N // f.dom.FG}


/-- Two structures `M` and `N` form an extension pair if the domain of any finitely-generated map
from `M` to `N` can be extended to include any element of `M`. -/
def IsExtensionPair : Prop := ∀ (f : L.FGEquiv M N) (m : M), ∃ g, m ∈ g.1.dom ∧ f ≤ g


theorem countable_self_fgequiv_of_countable [Countable M] :
    Countable (L.FGEquiv M M) := by
  let g : L.FGEquiv M M →
      Σ U : { S : L.Substructure M // S.FG }, U.val →[L] M :=
    fun f ↦ ⟨⟨f.val.dom, f.prop⟩, (subtype _).toHom.comp f.val.toEquiv.toHom⟩
  have g_inj : Function.Injective g := by
    intro f f' h
    ext
    let ⟨⟨dom_f, cod_f, equiv_f⟩, f_fin⟩ := f
    cases congr_arg (·.1) h
    apply PartialEquiv.ext (by rfl)
    simp only [g, Sigma.mk.inj_iff, heq_eq_eq, true_and] at h
    exact fun x hx ↦ congr_fun (congr_arg (↑) h) ⟨x, hx⟩
  have : ∀ U : { S : L.Substructure M // S.FG }, Structure.FG L U.val :=
    fun U ↦ (U.val.fg_iff_structure_fg.1 U.prop)
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    inst✝ : Countable M
    g : L.FGEquiv M M → Sigma fun U => L.Hom (Subtype fun x => Membership.mem (↑U) …
    g_inj : Function.Injective g
    this : ∀ (U : Subtype fun S => S.FG), FirstOrder.Language.Structure.FG L (Subt …
    ⊢ Countable (L.FGEquiv M M)
  -/
  exact Function.Embedding.countable ⟨g, g_inj⟩
  /-
    🎉 no goals
  -/


instance inhabited_self_FGEquiv : Inhabited (L.FGEquiv M M) :=
  ⟨⟨⟨⊥, ⊥, Equiv.refl L (⊥ : L.Substructure M)⟩, fg_bot⟩⟩


instance inhabited_FGEquiv_of_IsEmpty_Constants_and_Relations
    [IsEmpty L.Constants] [IsEmpty (L.Relations 0)] [L.Structure N] :
    Inhabited (L.FGEquiv M N) :=
  ⟨⟨⟨⊥, ⊥, {
      toFun := isEmptyElim
      invFun := isEmptyElim
      left_inv := isEmptyElim
      right_inv := isEmptyElim
      map_fun' := fun {n} f x => by
        /-
          L : FirstOrder.Language
          M : Type w
          N : Type w'
          inst✝⁴ : L.Structure M
          inst✝³ : L.Structure N
          inst✝² : IsEmpty L.Constants
          inst✝¹ : IsEmpty (L.Relations 0)
          inst✝ : L.Structure N
          n : Nat
          f : L.Functions n
          x : Fin n → Subtype fun x => Membership.mem Bot.bot x
          ⊢ Eq ({ toFun := fun a => isEmptyElim a, invFun := fun a => isEmptyElim a, lef …
        -/
        cases n
          /-
            case zero
            L : FirstOrder.Language
            M : Type w
            N : Type w'
            inst✝⁴ : L.Structure M
            inst✝³ : L.Structure N
            inst✝² : IsEmpty L.Constants
            inst✝¹ : IsEmpty (L.Relations 0)
            inst✝ : L.Structure N
            f : L.Functions 0
            x : Fin 0 → Subtype fun x => Membership.mem Bot.bot x
            ⊢ Eq ({ toFun := fun a => isEmptyElim a, invFun := fun a => isEmptyElim a, lef …
          -/
        · exact isEmptyElim f
          /-
            🎉 no goals
          -/
          /-
            case succ
            L : FirstOrder.Language
            M : Type w
            N : Type w'
            inst✝⁴ : L.Structure M
            inst✝³ : L.Structure N
            inst✝² : IsEmpty L.Constants
            inst✝¹ : IsEmpty (L.Relations 0)
            inst✝ : L.Structure N
            n✝ : Nat
            f : L.Functions (HAdd.hAdd n✝ 1)
            x : Fin (HAdd.hAdd n✝ 1) → Subtype fun x => Membership.mem Bot.bot x
            ⊢ Eq ({ toFun := fun a => isEmptyElim a, invFun := fun a => isEmptyElim a, lef …
          -/
        · exact isEmptyElim (x 0)
          /-
            🎉 no goals
          -/
      map_rel' := fun {n} r x => by
        /-
          L : FirstOrder.Language
          M : Type w
          N : Type w'
          inst✝⁴ : L.Structure M
          inst✝³ : L.Structure N
          inst✝² : IsEmpty L.Constants
          inst✝¹ : IsEmpty (L.Relations 0)
          inst✝ : L.Structure N
          n : Nat
          r : L.Relations n
          x : Fin n → Subtype fun x => Membership.mem Bot.bot x
          ⊢ Iff (FirstOrder.Language.Structure.RelMap r (Function.comp { toFun := fun a  …
        -/
        cases n
          /-
            case zero
            L : FirstOrder.Language
            M : Type w
            N : Type w'
            inst✝⁴ : L.Structure M
            inst✝³ : L.Structure N
            inst✝² : IsEmpty L.Constants
            inst✝¹ : IsEmpty (L.Relations 0)
            inst✝ : L.Structure N
            r : L.Relations 0
            x : Fin 0 → Subtype fun x => Membership.mem Bot.bot x
            ⊢ Iff (FirstOrder.Language.Structure.RelMap r (Function.comp { toFun := fun a  …
          -/
        · exact isEmptyElim r
          /-
            🎉 no goals
          -/
          /-
            case succ
            L : FirstOrder.Language
            M : Type w
            N : Type w'
            inst✝⁴ : L.Structure M
            inst✝³ : L.Structure N
            inst✝² : IsEmpty L.Constants
            inst✝¹ : IsEmpty (L.Relations 0)
            inst✝ : L.Structure N
            n✝ : Nat
            r : L.Relations (HAdd.hAdd n✝ 1)
            x : Fin (HAdd.hAdd n✝ 1) → Subtype fun x => Membership.mem Bot.bot x
            ⊢ Iff (FirstOrder.Language.Structure.RelMap r (Function.comp { toFun := fun a  …
          -/
        · exact isEmptyElim (x 0)
          /-
            🎉 no goals
          -/
    }⟩, fg_bot⟩⟩


/-- Maps to the symmetric finitely-generated partial equivalence. -/
@[simps]
def FGEquiv.symm (f : L.FGEquiv M N) : L.FGEquiv N M := ⟨f.1.symm, f.1.dom_fg_iff_cod_fg.1 f.2⟩


lemma isExtensionPair_iff_cod : L.IsExtensionPair M N ↔
    ∀ (f : L.FGEquiv N M) (m : M), ∃ g, m ∈ g.1.cod ∧ f ≤ g := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    ⊢ Iff (L.IsExtensionPair M N) (∀ (f : L.FGEquiv N M) (m : M), Exists fun g =>  …
  -/
  refine Iff.intro ?_ ?_ <;>
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      ⊢ L.IsExtensionPair M N → ∀ (f : L.FGEquiv N M) (m : M), Exists fun g => And ( …
    -/
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      h : L.IsExtensionPair M N
      f : L.FGEquiv N M
      m : M
      ⊢ Exists fun g => And (Membership.mem (↑g).cod m) (LE.le f g)
    -/
    /-
      case refine_1.intro.intro
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      h : L.IsExtensionPair M N
      f : L.FGEquiv N M
      m : M
      g : L.FGEquiv M N
      h1 : Membership.mem (↑g).dom m
      h2 : LE.le f.symm g
      ⊢ Exists fun g => And (Membership.mem (↑g).cod m) (LE.le f g)
    -/
    /-
      🎉 no goals
    -/
    obtain ⟨g, h1, h2⟩ := h f.symm m
    /-
      case refine_2.intro.intro
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      h : ∀ (f : L.FGEquiv N M) (m : M), Exists fun g => And (Membership.mem (↑g).co …
      f : L.FGEquiv M N
      m : M
      g : L.FGEquiv N M
      h1 : Membership.mem (↑g).cod m
      h2 : LE.le f.symm g
      ⊢ Exists fun g => And (Membership.mem (↑g).dom m) (LE.le f g)
    -/
    exact ⟨g.symm, h1, monotone_symm h2⟩
    /-
      🎉 no goals
    -/


/-- An alternate characterization of an extension pair is that every finitely generated partial
isomorphism can be extended to include any particular element of the domain. -/
theorem isExtensionPair_iff_exists_embedding_closure_singleton_sup :
    L.IsExtensionPair M N ↔
    ∀ (S : L.Substructure M) (_ : S.FG) (f : S ↪[L] N) (m : M),
      ∃ g : (closure L {m} ⊔ S : L.Substructure M) ↪[L] N, f =
        g.comp (Substructure.inclusion le_sup_right) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    ⊢ Iff (L.IsExtensionPair M N) (∀ (S : L.Substructure M), S.FG → ∀ (f : L.Embed …
  -/
  refine ⟨fun h S S_FG f m => ?_, fun h ⟨f, f_FG⟩ m => ?_⟩
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      h : L.IsExtensionPair M N
      S : L.Substructure M
      S_FG : S.FG
      f : L.Embedding (Subtype fun x => Membership.mem S x) N
      m : M
      ⊢ Exists fun g => Eq f (g.comp (FirstOrder.Language.Substructure.inclusion ⋯))
    -/
  · obtain ⟨⟨f', hf'⟩, mf', ff'1, ff'2⟩ := h ⟨⟨S, _, f.equivRange⟩, S_FG⟩ m
    /-
      case refine_1.intro.mk.intro.intro
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      h : L.IsExtensionPair M N
      S : L.Substructure M
      S_FG : S.FG
      f : L.Embedding (Subtype fun x => Membership.mem S x) N
      m : M
      f' : L.PartialEquiv M N
      hf' : f'.dom.FG
      mf' : Membership.mem (↑⟨f', hf'⟩).dom m
      ff'1 : LE.le (↑⟨{ dom := S, cod := f.toHom.range, toEquiv := f.equivRange }, S …
      ff'2 : Eq ((↑⟨f', hf'⟩).cod.subtype.comp ((↑⟨f', hf'⟩).toEquiv.toEmbedding.com …
      ⊢ Exists fun g => Eq f (g.comp (FirstOrder.Language.Substructure.inclusion ⋯))
    -/
    refine ⟨f'.toEmbedding.comp (Substructure.inclusion ?_), ?_⟩
    · simp only [sup_le_iff, ff'1, closure_le, singleton_subset_iff, SetLike.mem_coe, mf',
        and_self]
      /-
        case refine_1.intro.mk.intro.intro.refine_2
        L : FirstOrder.Language
        M : Type w
        N : Type w'
        inst✝¹ : L.Structure M
        inst✝ : L.Structure N
        h : L.IsExtensionPair M N
        S : L.Substructure M
        S_FG : S.FG
        f : L.Embedding (Subtype fun x => Membership.mem S x) N
        m : M
        f' : L.PartialEquiv M N
        hf' : f'.dom.FG
        mf' : Membership.mem (↑⟨f', hf'⟩).dom m
        ff'1 : LE.le (↑⟨{ dom := S, cod := f.toHom.range, toEquiv := f.equivRange }, S …
        ff'2 : Eq ((↑⟨f', hf'⟩).cod.subtype.comp ((↑⟨f', hf'⟩).toEquiv.toEmbedding.com …
        ⊢ Eq f ((f'.toEmbedding.comp (FirstOrder.Language.Substructure.inclusion ⋯)).c …
      -/
    · ext ⟨x, hx⟩
      /-
        case refine_1.intro.mk.intro.intro.refine_2.h.mk
        L : FirstOrder.Language
        M : Type w
        N : Type w'
        inst✝¹ : L.Structure M
        inst✝ : L.Structure N
        h : L.IsExtensionPair M N
        S : L.Substructure M
        S_FG : S.FG
        f : L.Embedding (Subtype fun x => Membership.mem S x) N
        m : M
        f' : L.PartialEquiv M N
        hf' : f'.dom.FG
        mf' : Membership.mem (↑⟨f', hf'⟩).dom m
        ff'1 : LE.le (↑⟨{ dom := S, cod := f.toHom.range, toEquiv := f.equivRange }, S …
        ff'2 : Eq ((↑⟨f', hf'⟩).cod.subtype.comp ((↑⟨f', hf'⟩).toEquiv.toEmbedding.com …
        x : M
        hx : Membership.mem S x
        ⊢ Eq (f ⟨x, hx⟩) (((f'.toEmbedding.comp (FirstOrder.Language.Substructure.incl …
      -/
      rw [Embedding.subtype_equivRange] at ff'2
      simp only [← ff'2, Embedding.comp_apply, Substructure.coe_inclusion, inclusion_mk,
        Equiv.coe_toEmbedding, coeSubtype, PartialEquiv.toEmbedding_apply]
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      h : ∀ (S : L.Substructure M), S.FG → ∀ (f : L.Embedding (Subtype fun x => Memb …
      x✝ : L.FGEquiv M N
      m : M
      f : L.PartialEquiv M N
      f_FG : f.dom.FG
      ⊢ Exists fun g => And (Membership.mem (↑g).dom m) (LE.le ⟨f, f_FG⟩ g)
    -/
  · obtain ⟨f', eq_f'⟩ := h f.dom f_FG f.toEmbedding m
    refine ⟨⟨⟨closure L {m} ⊔ f.dom, f'.toHom.range, f'.equivRange⟩,
      (fg_closure_singleton _).sup f_FG⟩,
      subset_closure.trans (le_sup_left : (closure L) {m} ≤ _) (mem_singleton m),
      ⟨le_sup_right, Embedding.ext (fun _ => ?_)⟩⟩
    /-
      case refine_2.intro
      L : FirstOrder.Language
      M : Type w
      N : Type w'
      inst✝¹ : L.Structure M
      inst✝ : L.Structure N
      h : ∀ (S : L.Substructure M), S.FG → ∀ (f : L.Embedding (Subtype fun x => Memb …
      x✝¹ : L.FGEquiv M N
      m : M
      f : L.PartialEquiv M N
      f_FG : f.dom.FG
      f' : L.Embedding (Subtype fun x => Membership.mem (Max.max ((FirstOrder.Langua …
      eq_f' : Eq f.toEmbedding (f'.comp (FirstOrder.Language.Substructure.inclusion  …
      x✝ : Subtype fun x => Membership.mem (↑⟨f, f_FG⟩).dom x
      ⊢ Eq (((↑⟨{ dom := Max.max ((FirstOrder.Language.Substructure.closure L).toFun …
    -/
    rw [PartialEquiv.toEmbedding] at eq_f'
    simp only [Embedding.comp_apply, Substructure.coe_inclusion, Equiv.coe_toEmbedding, coeSubtype,
      Embedding.equivRange_apply, eq_f']


protected alias ⟨cod, _⟩ := isExtensionPair_iff_cod


/-- The cofinal set of finite equivalences with a given element in their domain. -/
def definedAtLeft
    (h : L.IsExtensionPair M N) (m : M) : Order.Cofinal (FGEquiv L M N) where
  carrier := {f | m ∈ f.val.dom}
  isCofinal := fun f => h f m


/-- The cofinal set of finite equivalences with a given element in their codomain. -/
def definedAtRight
    (h : L.IsExtensionPair N M) (n : N) : Order.Cofinal (FGEquiv L M N) where
  carrier := {f | n ∈ f.val.cod}
  isCofinal := fun f => h.cod f n


/-- For a countably generated structure `M` and a structure `N`, if any partial equivalence
between finitely generated substructures can be extended to any element in the domain,
then there exists an embedding of `M` in `N`. -/
theorem embedding_from_cg (M_cg : Structure.CG L M) (g : L.FGEquiv M N)
    (H : L.IsExtensionPair M N) :
    ∃ f : M ↪[L] N, g ≤ f.toPartialEquiv := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    M_cg : FirstOrder.Language.Structure.CG L M
    g : L.FGEquiv M N
    H : L.IsExtensionPair M N
    ⊢ Exists fun f => LE.le (↑g) f.toPartialEquiv
  -/
  rcases M_cg with ⟨X, _, X_gen⟩
  /-
    case mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    H : L.IsExtensionPair M N
    X : Set M
    left✝ : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    ⊢ Exists fun f => LE.le (↑g) f.toPartialEquiv
  -/
  have _ : Countable (↑X : Type _) := by simpa only [countable_coe_iff]
  /-
    case mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    H : L.IsExtensionPair M N
    X : Set M
    left✝ : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    x✝ : Countable ↑X
    ⊢ Exists fun f => LE.le (↑g) f.toPartialEquiv
  -/
  have _ : Encodable (↑X : Type _) := Encodable.ofCountable _
  /-
    case mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    H : L.IsExtensionPair M N
    X : Set M
    left✝ : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    x✝¹ : Countable ↑X
    x✝ : Encodable ↑X
    ⊢ Exists fun f => LE.le (↑g) f.toPartialEquiv
  -/
  let D : X → Order.Cofinal (FGEquiv L M N) := fun x ↦ H.definedAtLeft x
  let S : ℕ →o M ≃ₚ[L] N :=
    ⟨Subtype.val ∘ (Order.sequenceOfCofinals g D),
      (Subtype.mono_coe _).comp (Order.sequenceOfCofinals.monotone _ _)⟩
  /-
    case mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    H : L.IsExtensionPair M N
    X : Set M
    left✝ : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    x✝¹ : Countable ↑X
    x✝ : Encodable ↑X
    D : ↑X → Order.Cofinal (L.FGEquiv M N) := fun x => H.definedAtLeft ↑x
    S : OrderHom Nat (L.PartialEquiv M N) := { toFun := Function.comp Subtype.val  …
    ⊢ Exists fun f => LE.le (↑g) f.toPartialEquiv
  -/
  let F := DirectLimit.partialEquivLimit S
  have _ : X ⊆ F.dom := by
    intro x hx
    have := Order.sequenceOfCofinals.encode_mem g D ⟨x, hx⟩
    exact dom_le_dom
      (le_partialEquivLimit S (Encodable.encode (⟨x, hx⟩ : X) + 1)) this
  /-
    case mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    H : L.IsExtensionPair M N
    X : Set M
    left✝ : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    x✝² : Countable ↑X
    x✝¹ : Encodable ↑X
    D : ↑X → Order.Cofinal (L.FGEquiv M N) := fun x => H.definedAtLeft ↑x
    S : OrderHom Nat (L.PartialEquiv M N) := { toFun := Function.comp Subtype.val  …
    F : L.PartialEquiv M N := FirstOrder.Language.DirectLimit.partialEquivLimit S
    x✝ : HasSubset.Subset X ↑F.dom
    ⊢ Exists fun f => LE.le (↑g) f.toPartialEquiv
  -/
  have isTop : F.dom = ⊤ := by rwa [← top_le_iff, ← X_gen, Substructure.closure_le]
  exact ⟨toEmbeddingOfEqTop isTop,
        by convert (le_partialEquivLimit S 0); apply Embedding.toPartialEquiv_toEmbedding⟩


/-- For two countably generated structure `M` and `N`, if any PartialEquiv
between finitely generated substructures can be extended to any element in the domain and to
any element in the codomain, then there exists an equivalence between `M` and `N`. -/
theorem equiv_between_cg (M_cg : Structure.CG L M) (N_cg : Structure.CG L N)
    (g : L.FGEquiv M N)
    (ext_dom : L.IsExtensionPair M N)
    (ext_cod : L.IsExtensionPair N M) :
    ∃ f : M ≃[L] N, g ≤ f.toEmbedding.toPartialEquiv := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    M_cg : FirstOrder.Language.Structure.CG L M
    N_cg : FirstOrder.Language.Structure.CG L N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    ⊢ Exists fun f => LE.le (↑g) f.toEmbedding.toPartialEquiv
  -/
  rcases M_cg with ⟨X, X_count, X_gen⟩
  /-
    case mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    N_cg : FirstOrder.Language.Structure.CG L N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    ⊢ Exists fun f => LE.le (↑g) f.toEmbedding.toPartialEquiv
  -/
  rcases N_cg with ⟨Y, Y_count, Y_gen⟩
  /-
    case mk.intro.intro.mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    Y : Set N
    Y_count : Y.Countable
    Y_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun Y) Top.top
    ⊢ Exists fun f => LE.le (↑g) f.toEmbedding.toPartialEquiv
  -/
  have _ : Countable (↑X : Type _) := by simpa only [countable_coe_iff]
  /-
    case mk.intro.intro.mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    Y : Set N
    Y_count : Y.Countable
    Y_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun Y) Top.top
    x✝ : Countable ↑X
    ⊢ Exists fun f => LE.le (↑g) f.toEmbedding.toPartialEquiv
  -/
  have _ : Encodable (↑X : Type _) := Encodable.ofCountable _
  /-
    case mk.intro.intro.mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    Y : Set N
    Y_count : Y.Countable
    Y_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun Y) Top.top
    x✝¹ : Countable ↑X
    x✝ : Encodable ↑X
    ⊢ Exists fun f => LE.le (↑g) f.toEmbedding.toPartialEquiv
  -/
  have _ : Countable (↑Y : Type _) := by simpa only [countable_coe_iff]
  /-
    case mk.intro.intro.mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    Y : Set N
    Y_count : Y.Countable
    Y_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun Y) Top.top
    x✝² : Countable ↑X
    x✝¹ : Encodable ↑X
    x✝ : Countable ↑Y
    ⊢ Exists fun f => LE.le (↑g) f.toEmbedding.toPartialEquiv
  -/
  have _ : Encodable (↑Y : Type _) := Encodable.ofCountable _
  let D : Sum X Y → Order.Cofinal (FGEquiv L M N) := fun p ↦
    Sum.recOn p (fun x ↦ ext_dom.definedAtLeft x) (fun y ↦ ext_cod.definedAtRight y)
  let S : ℕ →o M ≃ₚ[L] N :=
    ⟨Subtype.val ∘ (Order.sequenceOfCofinals g D),
      (Subtype.mono_coe _).comp (Order.sequenceOfCofinals.monotone _ _)⟩
  /-
    case mk.intro.intro.mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    Y : Set N
    Y_count : Y.Countable
    Y_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun Y) Top.top
    x✝³ : Countable ↑X
    x✝² : Encodable ↑X
    x✝¹ : Countable ↑Y
    x✝ : Encodable ↑Y
    D : Sum ↑X ↑Y → Order.Cofinal (L.FGEquiv M N) := fun p => Sum.recOn p (fun x = …
    S : OrderHom Nat (L.PartialEquiv M N) := { toFun := Function.comp Subtype.val  …
    ⊢ Exists fun f => LE.le (↑g) f.toEmbedding.toPartialEquiv
  -/
  let F := @DirectLimit.partialEquivLimit L M N _ _ ℕ _ _ _ S
  have _ : X ⊆ F.dom := by
    intro x hx
    have := Order.sequenceOfCofinals.encode_mem g D (Sum.inl ⟨x, hx⟩)
    exact dom_le_dom
      (le_partialEquivLimit S (Encodable.encode (Sum.inl (⟨x, hx⟩ : X)) + 1)) this
  have _ : Y ⊆ F.cod := by
    intro y hy
    have := Order.sequenceOfCofinals.encode_mem g D (Sum.inr ⟨y, hy⟩)
    exact cod_le_cod
      (le_partialEquivLimit S (Encodable.encode (Sum.inr (⟨y, hy⟩ : Y)) + 1)) this
  /-
    case mk.intro.intro.mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    Y : Set N
    Y_count : Y.Countable
    Y_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun Y) Top.top
    x✝⁵ : Countable ↑X
    x✝⁴ : Encodable ↑X
    x✝³ : Countable ↑Y
    x✝² : Encodable ↑Y
    D : Sum ↑X ↑Y → Order.Cofinal (L.FGEquiv M N) := fun p => Sum.recOn p (fun x = …
    S : OrderHom Nat (L.PartialEquiv M N) := { toFun := Function.comp Subtype.val  …
    F : L.PartialEquiv M N := FirstOrder.Language.DirectLimit.partialEquivLimit S
    x✝¹ : HasSubset.Subset X ↑F.dom
    x✝ : HasSubset.Subset Y ↑F.cod
    ⊢ Exists fun f => LE.le (↑g) f.toEmbedding.toPartialEquiv
  -/
  have dom_top : F.dom = ⊤ := by rwa [← top_le_iff, ← X_gen, Substructure.closure_le]
  /-
    case mk.intro.intro.mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    Y : Set N
    Y_count : Y.Countable
    Y_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun Y) Top.top
    x✝⁵ : Countable ↑X
    x✝⁴ : Encodable ↑X
    x✝³ : Countable ↑Y
    x✝² : Encodable ↑Y
    D : Sum ↑X ↑Y → Order.Cofinal (L.FGEquiv M N) := fun p => Sum.recOn p (fun x = …
    S : OrderHom Nat (L.PartialEquiv M N) := { toFun := Function.comp Subtype.val  …
    F : L.PartialEquiv M N := FirstOrder.Language.DirectLimit.partialEquivLimit S
    x✝¹ : HasSubset.Subset X ↑F.dom
    x✝ : HasSubset.Subset Y ↑F.cod
    dom_top : Eq F.dom Top.top
    ⊢ Exists fun f => LE.le (↑g) f.toEmbedding.toPartialEquiv
  -/
  have cod_top : F.cod = ⊤ := by rwa [← top_le_iff, ← Y_gen, Substructure.closure_le]
  /-
    case mk.intro.intro.mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    Y : Set N
    Y_count : Y.Countable
    Y_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun Y) Top.top
    x✝⁵ : Countable ↑X
    x✝⁴ : Encodable ↑X
    x✝³ : Countable ↑Y
    x✝² : Encodable ↑Y
    D : Sum ↑X ↑Y → Order.Cofinal (L.FGEquiv M N) := fun p => Sum.recOn p (fun x = …
    S : OrderHom Nat (L.PartialEquiv M N) := { toFun := Function.comp Subtype.val  …
    F : L.PartialEquiv M N := FirstOrder.Language.DirectLimit.partialEquivLimit S
    x✝¹ : HasSubset.Subset X ↑F.dom
    x✝ : HasSubset.Subset Y ↑F.cod
    dom_top : Eq F.dom Top.top
    cod_top : Eq F.cod Top.top
    ⊢ Exists fun f => LE.le (↑g) f.toEmbedding.toPartialEquiv
  -/
  refine ⟨toEquivOfEqTop dom_top cod_top, ?_⟩
  /-
    case mk.intro.intro.mk.intro.intro
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    Y : Set N
    Y_count : Y.Countable
    Y_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun Y) Top.top
    x✝⁵ : Countable ↑X
    x✝⁴ : Encodable ↑X
    x✝³ : Countable ↑Y
    x✝² : Encodable ↑Y
    D : Sum ↑X ↑Y → Order.Cofinal (L.FGEquiv M N) := fun p => Sum.recOn p (fun x = …
    S : OrderHom Nat (L.PartialEquiv M N) := { toFun := Function.comp Subtype.val  …
    F : L.PartialEquiv M N := FirstOrder.Language.DirectLimit.partialEquivLimit S
    x✝¹ : HasSubset.Subset X ↑F.dom
    x✝ : HasSubset.Subset Y ↑F.cod
    dom_top : Eq F.dom Top.top
    cod_top : Eq F.cod Top.top
    ⊢ LE.le (↑g) (FirstOrder.Language.PartialEquiv.toEquivOfEqTop dom_top cod_top) …
  -/
  convert le_partialEquivLimit S 0
  /-
    case h.e'_4
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    Y : Set N
    Y_count : Y.Countable
    Y_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun Y) Top.top
    x✝⁵ : Countable ↑X
    x✝⁴ : Encodable ↑X
    x✝³ : Countable ↑Y
    x✝² : Encodable ↑Y
    D : Sum ↑X ↑Y → Order.Cofinal (L.FGEquiv M N) := fun p => Sum.recOn p (fun x = …
    S : OrderHom Nat (L.PartialEquiv M N) := { toFun := Function.comp Subtype.val  …
    F : L.PartialEquiv M N := FirstOrder.Language.DirectLimit.partialEquivLimit S
    x✝¹ : HasSubset.Subset X ↑F.dom
    x✝ : HasSubset.Subset Y ↑F.cod
    dom_top : Eq F.dom Top.top
    cod_top : Eq F.cod Top.top
    ⊢ Eq (FirstOrder.Language.PartialEquiv.toEquivOfEqTop dom_top cod_top).toEmbed …
  -/
  rw [toEquivOfEqTop_toEmbedding]
  /-
    case h.e'_4
    L : FirstOrder.Language
    M : Type w
    N : Type w'
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    g : L.FGEquiv M N
    ext_dom : L.IsExtensionPair M N
    ext_cod : L.IsExtensionPair N M
    X : Set M
    X_count : X.Countable
    X_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun X) Top.top
    Y : Set N
    Y_count : Y.Countable
    Y_gen : Eq ((FirstOrder.Language.Substructure.closure L).toFun Y) Top.top
    x✝⁵ : Countable ↑X
    x✝⁴ : Encodable ↑X
    x✝³ : Countable ↑Y
    x✝² : Encodable ↑Y
    D : Sum ↑X ↑Y → Order.Cofinal (L.FGEquiv M N) := fun p => Sum.recOn p (fun x = …
    S : OrderHom Nat (L.PartialEquiv M N) := { toFun := Function.comp Subtype.val  …
    F : L.PartialEquiv M N := FirstOrder.Language.DirectLimit.partialEquivLimit S
    x✝¹ : HasSubset.Subset X ↑F.dom
    x✝ : HasSubset.Subset Y ↑F.cod
    dom_top : Eq F.dom Top.top
    cod_top : Eq F.cod Top.top
    ⊢ Eq (FirstOrder.Language.PartialEquiv.toEmbeddingOfEqTop dom_top).toPartialEq …
  -/
  apply Embedding.toPartialEquiv_toEmbedding
  /-
    🎉 no goals
  -/


