/-- We call a category `NonPreadditiveAbelian` if it has a zero object, kernels, cokernels, binary
    products and coproducts, and every monomorphism and every epimorphism is normal. -/
class NonPreadditiveAbelian extends HasZeroMorphisms C, NormalMonoCategory C,
    NormalEpiCategory C where
  [has_zero_object : HasZeroObject C]
  [has_kernels : HasKernels C]
  [has_cokernels : HasCokernels C]
  [has_finite_products : HasFiniteProducts C]
  [has_finite_coproducts : HasFiniteCoproducts C]


/-- The map `p : P ⟶ image f` is an epimorphism -/
instance : Epi (Abelian.factorThruImage f) :=
  let I := Abelian.image f
  let p := Abelian.factorThruImage f
  let i := kernel.ι (cokernel.π f)
  -- It will suffice to consider some g : I ⟶ R such that p ≫ g = 0 and show that g = 0.
  NormalMonoCategory.epi_of_zero_cancel
  _ fun R (g : I ⟶ R) (hpg : p ≫ g = 0) => by
  -- Since C is abelian, u := ker g ≫ i is the kernel of some morphism h.
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    P Q : C
    f : Quiver.Hom P Q
    I : C := CategoryTheory.Abelian.image f
    p : Quiver.Hom P (CategoryTheory.Abelian.image f) := CategoryTheory.Abelian.fa …
    i : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    R : C
    g : Quiver.Hom I R
    hpg : Eq (CategoryTheory.CategoryStruct.comp p g) 0
    ⊢ Eq g 0
  -/
  let u := kernel.ι g ≫ i
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    P Q : C
    f : Quiver.Hom P Q
    I : C := CategoryTheory.Abelian.image f
    p : Quiver.Hom P (CategoryTheory.Abelian.image f) := CategoryTheory.Abelian.fa …
    i : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    R : C
    g : Quiver.Hom I R
    hpg : Eq (CategoryTheory.CategoryStruct.comp p g) 0
    u : Quiver.Hom (CategoryTheory.Limits.kernel g) Q := CategoryTheory.CategorySt …
    ⊢ Eq g 0
  -/
  haveI hu := normalMonoOfMono u
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    P Q : C
    f : Quiver.Hom P Q
    I : C := CategoryTheory.Abelian.image f
    p : Quiver.Hom P (CategoryTheory.Abelian.image f) := CategoryTheory.Abelian.fa …
    i : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    R : C
    g : Quiver.Hom I R
    hpg : Eq (CategoryTheory.CategoryStruct.comp p g) 0
    u : Quiver.Hom (CategoryTheory.Limits.kernel g) Q := CategoryTheory.CategorySt …
    hu : CategoryTheory.NormalMono u
    ⊢ Eq g 0
  -/
  let h := hu.g
  -- By hypothesis, p factors through the kernel of g via some t.
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    P Q : C
    f : Quiver.Hom P Q
    I : C := CategoryTheory.Abelian.image f
    p : Quiver.Hom P (CategoryTheory.Abelian.image f) := CategoryTheory.Abelian.fa …
    i : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    R : C
    g : Quiver.Hom I R
    hpg : Eq (CategoryTheory.CategoryStruct.comp p g) 0
    u : Quiver.Hom (CategoryTheory.Limits.kernel g) Q := CategoryTheory.CategorySt …
    hu : CategoryTheory.NormalMono u
    h : Quiver.Hom Q (CategoryTheory.NormalMono.Z u) := CategoryTheory.NormalMono.g
    ⊢ Eq g 0
  -/
  obtain ⟨t, ht⟩ := kernel.lift' g p hpg
  have fh : f ≫ h = 0 :=
    calc
      f ≫ h = (p ≫ i) ≫ h := (Abelian.image.fac f).symm ▸ rfl
      _ = ((t ≫ kernel.ι g) ≫ i) ≫ h := ht ▸ rfl
      _ = t ≫ u ≫ h := by simp only [u, Category.assoc]
      _ = t ≫ 0 := hu.w ▸ rfl
      _ = 0 := HasZeroMorphisms.comp_zero _ _
  -- h factors through the cokernel of f via some l.
  /-
    case mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    P Q : C
    f : Quiver.Hom P Q
    I : C := CategoryTheory.Abelian.image f
    p : Quiver.Hom P (CategoryTheory.Abelian.image f) := CategoryTheory.Abelian.fa …
    i : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    R : C
    g : Quiver.Hom I R
    hpg : Eq (CategoryTheory.CategoryStruct.comp p g) 0
    u : Quiver.Hom (CategoryTheory.Limits.kernel g) Q := CategoryTheory.CategorySt …
    hu : CategoryTheory.NormalMono u
    h : Quiver.Hom Q (CategoryTheory.NormalMono.Z u) := CategoryTheory.NormalMono.g
    t : Quiver.Hom P (CategoryTheory.Limits.kernel g)
    ht : Eq (CategoryTheory.CategoryStruct.comp t (CategoryTheory.Limits.kernel.ι  …
    fh : Eq (CategoryTheory.CategoryStruct.comp f h) 0
    ⊢ Eq g 0
  -/
  obtain ⟨l, hl⟩ := cokernel.desc' f h fh
  have hih : i ≫ h = 0 :=
    calc
      i ≫ h = i ≫ cokernel.π f ≫ l := hl ▸ rfl
      _ = 0 ≫ l := by rw [← Category.assoc, kernel.condition]
      _ = 0 := zero_comp
  -- i factors through u = ker h via some s.
  /-
    case mk.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    P Q : C
    f : Quiver.Hom P Q
    I : C := CategoryTheory.Abelian.image f
    p : Quiver.Hom P (CategoryTheory.Abelian.image f) := CategoryTheory.Abelian.fa …
    i : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    R : C
    g : Quiver.Hom I R
    hpg : Eq (CategoryTheory.CategoryStruct.comp p g) 0
    u : Quiver.Hom (CategoryTheory.Limits.kernel g) Q := CategoryTheory.CategorySt …
    hu : CategoryTheory.NormalMono u
    h : Quiver.Hom Q (CategoryTheory.NormalMono.Z u) := CategoryTheory.NormalMono.g
    t : Quiver.Hom P (CategoryTheory.Limits.kernel g)
    ht : Eq (CategoryTheory.CategoryStruct.comp t (CategoryTheory.Limits.kernel.ι  …
    fh : Eq (CategoryTheory.CategoryStruct.comp f h) 0
    l : Quiver.Hom (CategoryTheory.Limits.cokernel f) (CategoryTheory.NormalMono.Z …
    hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π  …
    hih : Eq (CategoryTheory.CategoryStruct.comp i h) 0
    ⊢ Eq g 0
  -/
  obtain ⟨s, hs⟩ := NormalMono.lift' u i hih
  /-
    case mk.mk.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    P Q : C
    f : Quiver.Hom P Q
    I : C := CategoryTheory.Abelian.image f
    p : Quiver.Hom P (CategoryTheory.Abelian.image f) := CategoryTheory.Abelian.fa …
    i : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    R : C
    g : Quiver.Hom I R
    hpg : Eq (CategoryTheory.CategoryStruct.comp p g) 0
    u : Quiver.Hom (CategoryTheory.Limits.kernel g) Q := CategoryTheory.CategorySt …
    hu : CategoryTheory.NormalMono u
    h : Quiver.Hom Q (CategoryTheory.NormalMono.Z u) := CategoryTheory.NormalMono.g
    t : Quiver.Hom P (CategoryTheory.Limits.kernel g)
    ht : Eq (CategoryTheory.CategoryStruct.comp t (CategoryTheory.Limits.kernel.ι  …
    fh : Eq (CategoryTheory.CategoryStruct.comp f h) 0
    l : Quiver.Hom (CategoryTheory.Limits.cokernel f) (CategoryTheory.NormalMono.Z …
    hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π  …
    hih : Eq (CategoryTheory.CategoryStruct.comp i h) 0
    s : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    hs : Eq (CategoryTheory.CategoryStruct.comp s u) i
    ⊢ Eq g 0
  -/
  have hs' : (s ≫ kernel.ι g) ≫ i = 𝟙 I ≫ i := by rw [Category.assoc, hs, Category.id_comp]
  /-
    case mk.mk.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    P Q : C
    f : Quiver.Hom P Q
    I : C := CategoryTheory.Abelian.image f
    p : Quiver.Hom P (CategoryTheory.Abelian.image f) := CategoryTheory.Abelian.fa …
    i : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    R : C
    g : Quiver.Hom I R
    hpg : Eq (CategoryTheory.CategoryStruct.comp p g) 0
    u : Quiver.Hom (CategoryTheory.Limits.kernel g) Q := CategoryTheory.CategorySt …
    hu : CategoryTheory.NormalMono u
    h : Quiver.Hom Q (CategoryTheory.NormalMono.Z u) := CategoryTheory.NormalMono.g
    t : Quiver.Hom P (CategoryTheory.Limits.kernel g)
    ht : Eq (CategoryTheory.CategoryStruct.comp t (CategoryTheory.Limits.kernel.ι  …
    fh : Eq (CategoryTheory.CategoryStruct.comp f h) 0
    l : Quiver.Hom (CategoryTheory.Limits.cokernel f) (CategoryTheory.NormalMono.Z …
    hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π  …
    hih : Eq (CategoryTheory.CategoryStruct.comp i h) 0
    s : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    hs : Eq (CategoryTheory.CategoryStruct.comp s u) i
    hs' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    ⊢ Eq g 0
  -/
  haveI : Epi (kernel.ι g) := epi_of_epi_fac ((cancel_mono _).1 hs')
  -- ker g is an epimorphism, but ker g ≫ g = 0 = ker g ≫ 0, so g = 0 as required.
  /-
    case mk.mk.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    P Q : C
    f : Quiver.Hom P Q
    I : C := CategoryTheory.Abelian.image f
    p : Quiver.Hom P (CategoryTheory.Abelian.image f) := CategoryTheory.Abelian.fa …
    i : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    R : C
    g : Quiver.Hom I R
    hpg : Eq (CategoryTheory.CategoryStruct.comp p g) 0
    u : Quiver.Hom (CategoryTheory.Limits.kernel g) Q := CategoryTheory.CategorySt …
    hu : CategoryTheory.NormalMono u
    h : Quiver.Hom Q (CategoryTheory.NormalMono.Z u) := CategoryTheory.NormalMono.g
    t : Quiver.Hom P (CategoryTheory.Limits.kernel g)
    ht : Eq (CategoryTheory.CategoryStruct.comp t (CategoryTheory.Limits.kernel.ι  …
    fh : Eq (CategoryTheory.CategoryStruct.comp f h) 0
    l : Quiver.Hom (CategoryTheory.Limits.cokernel f) (CategoryTheory.NormalMono.Z …
    hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π  …
    hih : Eq (CategoryTheory.CategoryStruct.comp i h) 0
    s : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
    hs : Eq (CategoryTheory.CategoryStruct.comp s u) i
    hs' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    this : CategoryTheory.Epi (CategoryTheory.Limits.kernel.ι g)
    ⊢ Eq g 0
  -/
  exact zero_of_epi_comp _ (kernel.condition g)
  /-
    🎉 no goals
  -/


instance isIso_factorThruImage [Mono f] : IsIso (Abelian.factorThruImage f) :=
  isIso_of_mono_of_epi <| Abelian.factorThruImage f


/-- The canonical morphism `i : coimage f ⟶ Q` is a monomorphism -/
instance : Mono (Abelian.factorThruCoimage f) :=
  let I := Abelian.coimage f
  let i := Abelian.factorThruCoimage f
  let p := cokernel.π (kernel.ι f)
  NormalEpiCategory.mono_of_cancel_zero _ fun R (g : R ⟶ I) (hgi : g ≫ i = 0) => by
    -- Since C is abelian, u := p ≫ coker g is the cokernel of some morphism h.
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.NonPreadditiveAbelian C
      P Q : C
      f : Quiver.Hom P Q
      I : C := CategoryTheory.Abelian.coimage f
      i : Quiver.Hom (CategoryTheory.Abelian.coimage f) Q := CategoryTheory.Abelian. …
      p : Quiver.Hom P (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.kernel …
      R : C
      g : Quiver.Hom R I
      hgi : Eq (CategoryTheory.CategoryStruct.comp g i) 0
      ⊢ Eq g 0
    -/
    let u := p ≫ cokernel.π g
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.NonPreadditiveAbelian C
      P Q : C
      f : Quiver.Hom P Q
      I : C := CategoryTheory.Abelian.coimage f
      i : Quiver.Hom (CategoryTheory.Abelian.coimage f) Q := CategoryTheory.Abelian. …
      p : Quiver.Hom P (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.kernel …
      R : C
      g : Quiver.Hom R I
      hgi : Eq (CategoryTheory.CategoryStruct.comp g i) 0
      u : Quiver.Hom P (CategoryTheory.Limits.cokernel g) := CategoryTheory.Category …
      ⊢ Eq g 0
    -/
    haveI hu := normalEpiOfEpi u
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.NonPreadditiveAbelian C
      P Q : C
      f : Quiver.Hom P Q
      I : C := CategoryTheory.Abelian.coimage f
      i : Quiver.Hom (CategoryTheory.Abelian.coimage f) Q := CategoryTheory.Abelian. …
      p : Quiver.Hom P (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.kernel …
      R : C
      g : Quiver.Hom R I
      hgi : Eq (CategoryTheory.CategoryStruct.comp g i) 0
      u : Quiver.Hom P (CategoryTheory.Limits.cokernel g) := CategoryTheory.Category …
      hu : CategoryTheory.NormalEpi u
      ⊢ Eq g 0
    -/
    let h := hu.g
    -- By hypothesis, i factors through the cokernel of g via some t.
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.NonPreadditiveAbelian C
      P Q : C
      f : Quiver.Hom P Q
      I : C := CategoryTheory.Abelian.coimage f
      i : Quiver.Hom (CategoryTheory.Abelian.coimage f) Q := CategoryTheory.Abelian. …
      p : Quiver.Hom P (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.kernel …
      R : C
      g : Quiver.Hom R I
      hgi : Eq (CategoryTheory.CategoryStruct.comp g i) 0
      u : Quiver.Hom P (CategoryTheory.Limits.cokernel g) := CategoryTheory.Category …
      hu : CategoryTheory.NormalEpi u
      h : Quiver.Hom (CategoryTheory.NormalEpi.W u) P := CategoryTheory.NormalEpi.g
      ⊢ Eq g 0
    -/
    obtain ⟨t, ht⟩ := cokernel.desc' g i hgi
    have hf : h ≫ f = 0 :=
      calc
        h ≫ f = h ≫ p ≫ i := (Abelian.coimage.fac f).symm ▸ rfl
        _ = h ≫ p ≫ cokernel.π g ≫ t := ht ▸ rfl
        _ = h ≫ u ≫ t := by simp only [u, Category.assoc]
        _ = 0 ≫ t := by rw [← Category.assoc, hu.w]
        _ = 0 := zero_comp
    -- h factors through the kernel of f via some l.
    /-
      case mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.NonPreadditiveAbelian C
      P Q : C
      f : Quiver.Hom P Q
      I : C := CategoryTheory.Abelian.coimage f
      i : Quiver.Hom (CategoryTheory.Abelian.coimage f) Q := CategoryTheory.Abelian. …
      p : Quiver.Hom P (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.kernel …
      R : C
      g : Quiver.Hom R I
      hgi : Eq (CategoryTheory.CategoryStruct.comp g i) 0
      u : Quiver.Hom P (CategoryTheory.Limits.cokernel g) := CategoryTheory.Category …
      hu : CategoryTheory.NormalEpi u
      h : Quiver.Hom (CategoryTheory.NormalEpi.W u) P := CategoryTheory.NormalEpi.g
      t : Quiver.Hom (CategoryTheory.Limits.cokernel g) Q
      ht : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π  …
      hf : Eq (CategoryTheory.CategoryStruct.comp h f) 0
      ⊢ Eq g 0
    -/
    obtain ⟨l, hl⟩ := kernel.lift' f h hf
    have hhp : h ≫ p = 0 :=
      calc
        h ≫ p = (l ≫ kernel.ι f) ≫ p := hl ▸ rfl
        _ = l ≫ 0 := by rw [Category.assoc, cokernel.condition]
        _ = 0 := comp_zero
    -- p factors through u = coker h via some s.
    /-
      case mk.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.NonPreadditiveAbelian C
      P Q : C
      f : Quiver.Hom P Q
      I : C := CategoryTheory.Abelian.coimage f
      i : Quiver.Hom (CategoryTheory.Abelian.coimage f) Q := CategoryTheory.Abelian. …
      p : Quiver.Hom P (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.kernel …
      R : C
      g : Quiver.Hom R I
      hgi : Eq (CategoryTheory.CategoryStruct.comp g i) 0
      u : Quiver.Hom P (CategoryTheory.Limits.cokernel g) := CategoryTheory.Category …
      hu : CategoryTheory.NormalEpi u
      h : Quiver.Hom (CategoryTheory.NormalEpi.W u) P := CategoryTheory.NormalEpi.g
      t : Quiver.Hom (CategoryTheory.Limits.cokernel g) Q
      ht : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π  …
      hf : Eq (CategoryTheory.CategoryStruct.comp h f) 0
      l : Quiver.Hom (CategoryTheory.NormalEpi.W u) (CategoryTheory.Limits.kernel f)
      hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.kernel.ι  …
      hhp : Eq (CategoryTheory.CategoryStruct.comp h p) 0
      ⊢ Eq g 0
    -/
    obtain ⟨s, hs⟩ := NormalEpi.desc' u p hhp
    /-
      case mk.mk.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.NonPreadditiveAbelian C
      P Q : C
      f : Quiver.Hom P Q
      I : C := CategoryTheory.Abelian.coimage f
      i : Quiver.Hom (CategoryTheory.Abelian.coimage f) Q := CategoryTheory.Abelian. …
      p : Quiver.Hom P (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.kernel …
      R : C
      g : Quiver.Hom R I
      hgi : Eq (CategoryTheory.CategoryStruct.comp g i) 0
      u : Quiver.Hom P (CategoryTheory.Limits.cokernel g) := CategoryTheory.Category …
      hu : CategoryTheory.NormalEpi u
      h : Quiver.Hom (CategoryTheory.NormalEpi.W u) P := CategoryTheory.NormalEpi.g
      t : Quiver.Hom (CategoryTheory.Limits.cokernel g) Q
      ht : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π  …
      hf : Eq (CategoryTheory.CategoryStruct.comp h f) 0
      l : Quiver.Hom (CategoryTheory.NormalEpi.W u) (CategoryTheory.Limits.kernel f)
      hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.kernel.ι  …
      hhp : Eq (CategoryTheory.CategoryStruct.comp h p) 0
      s : Quiver.Hom (CategoryTheory.Limits.cokernel g) (CategoryTheory.Limits.coker …
      hs : Eq (CategoryTheory.CategoryStruct.comp u s) p
      ⊢ Eq g 0
    -/
    have hs' : p ≫ cokernel.π g ≫ s = p ≫ 𝟙 I := by rw [← Category.assoc, hs, Category.comp_id]
    /-
      case mk.mk.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.NonPreadditiveAbelian C
      P Q : C
      f : Quiver.Hom P Q
      I : C := CategoryTheory.Abelian.coimage f
      i : Quiver.Hom (CategoryTheory.Abelian.coimage f) Q := CategoryTheory.Abelian. …
      p : Quiver.Hom P (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.kernel …
      R : C
      g : Quiver.Hom R I
      hgi : Eq (CategoryTheory.CategoryStruct.comp g i) 0
      u : Quiver.Hom P (CategoryTheory.Limits.cokernel g) := CategoryTheory.Category …
      hu : CategoryTheory.NormalEpi u
      h : Quiver.Hom (CategoryTheory.NormalEpi.W u) P := CategoryTheory.NormalEpi.g
      t : Quiver.Hom (CategoryTheory.Limits.cokernel g) Q
      ht : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π  …
      hf : Eq (CategoryTheory.CategoryStruct.comp h f) 0
      l : Quiver.Hom (CategoryTheory.NormalEpi.W u) (CategoryTheory.Limits.kernel f)
      hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.kernel.ι  …
      hhp : Eq (CategoryTheory.CategoryStruct.comp h p) 0
      s : Quiver.Hom (CategoryTheory.Limits.cokernel g) (CategoryTheory.Limits.coker …
      hs : Eq (CategoryTheory.CategoryStruct.comp u s) p
      hs' : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.CategoryStruct. …
      ⊢ Eq g 0
    -/
    haveI : Mono (cokernel.π g) := mono_of_mono_fac ((cancel_epi _).1 hs')
    -- coker g is a monomorphism, but g ≫ coker g = 0 = 0 ≫ coker g, so g = 0 as required.
    /-
      case mk.mk.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.NonPreadditiveAbelian C
      P Q : C
      f : Quiver.Hom P Q
      I : C := CategoryTheory.Abelian.coimage f
      i : Quiver.Hom (CategoryTheory.Abelian.coimage f) Q := CategoryTheory.Abelian. …
      p : Quiver.Hom P (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.kernel …
      R : C
      g : Quiver.Hom R I
      hgi : Eq (CategoryTheory.CategoryStruct.comp g i) 0
      u : Quiver.Hom P (CategoryTheory.Limits.cokernel g) := CategoryTheory.Category …
      hu : CategoryTheory.NormalEpi u
      h : Quiver.Hom (CategoryTheory.NormalEpi.W u) P := CategoryTheory.NormalEpi.g
      t : Quiver.Hom (CategoryTheory.Limits.cokernel g) Q
      ht : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π  …
      hf : Eq (CategoryTheory.CategoryStruct.comp h f) 0
      l : Quiver.Hom (CategoryTheory.NormalEpi.W u) (CategoryTheory.Limits.kernel f)
      hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.kernel.ι  …
      hhp : Eq (CategoryTheory.CategoryStruct.comp h p) 0
      s : Quiver.Hom (CategoryTheory.Limits.cokernel g) (CategoryTheory.Limits.coker …
      hs : Eq (CategoryTheory.CategoryStruct.comp u s) p
      hs' : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.CategoryStruct. …
      this : CategoryTheory.Mono (CategoryTheory.Limits.cokernel.π g)
      ⊢ Eq g 0
    -/
    exact zero_of_comp_mono _ (cokernel.condition g)
    /-
      🎉 no goals
    -/


instance isIso_factorThruCoimage [Epi f] : IsIso (Abelian.factorThruCoimage f) :=
  isIso_of_mono_of_epi _


/-- In a `NonPreadditiveAbelian` category, an epi is the cokernel of its kernel. More precisely:
    If `f` is an epimorphism and `s` is some limit kernel cone on `f`, then `f` is a cokernel
    of `Fork.ι s`. -/
def epiIsCokernelOfKernel [Epi f] (s : Fork f 0) (h : IsLimit s) :
    IsColimit (CokernelCofork.ofπ f (KernelFork.condition s)) :=
  IsCokernel.cokernelIso _ _
    (cokernel.ofIsoComp _ _ (Limits.IsLimit.conePointUniqueUpToIso (limit.isLimit _) h)
      (ConeMorphism.w (Limits.IsLimit.uniqueUpToIso (limit.isLimit _) h).hom _))
    (asIso <| Abelian.factorThruCoimage f) (Abelian.coimage.fac f)


/-- In a `NonPreadditiveAbelian` category, a mono is the kernel of its cokernel. More precisely:
    If `f` is a monomorphism and `s` is some colimit cokernel cocone on `f`, then `f` is a kernel
    of `Cofork.π s`. -/
def monoIsKernelOfCokernel [Mono f] (s : Cofork f 0) (h : IsColimit s) :
    IsLimit (KernelFork.ofι f (CokernelCofork.condition s)) :=
  IsKernel.isoKernel _ _
    (kernel.ofCompIso _ _ (Limits.IsColimit.coconePointUniqueUpToIso h (colimit.isColimit _))
      (CoconeMorphism.w (Limits.IsColimit.uniqueUpToIso h <| colimit.isColimit _).hom _))
    (asIso <| Abelian.factorThruImage f) (Abelian.image.fac f)


/-- The composite `A ⟶ A ⨯ A ⟶ cokernel (Δ A)`, where the first map is `(𝟙 A, 0)` and the second map
    is the canonical projection into the cokernel. -/
abbrev r (A : C) : A ⟶ cokernel (diag A) :=
  prod.lift (𝟙 A) 0 ≫ cokernel.π (diag A)


instance mono_Δ {A : C} : Mono (diag A) :=
  mono_of_mono_fac <| prod.lift_fst _ _


instance mono_r {A : C} : Mono (r A) := by
  let hl : IsLimit (KernelFork.ofι (diag A) (cokernel.condition (diag A))) :=
    monoIsKernelOfCokernel _ (colimit.isColimit _)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
    ⊢ CategoryTheory.Mono (CategoryTheory.NonPreadditiveAbelian.r A)
  -/
  apply NormalEpiCategory.mono_of_cancel_zero
  /-
    case hf
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
    ⊢ ∀ (Z : C) (g : Quiver.Hom Z A), Eq (CategoryTheory.CategoryStruct.comp g (Ca …
  -/
  intro Z x hx
  have hxx : (x ≫ prod.lift (𝟙 A) (0 : A ⟶ A)) ≫ cokernel.π (diag A) = 0 := by
    rw [Category.assoc, hx]
  /-
    case hf
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
    Z : C
    x : Quiver.Hom Z A
    hx : Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.NonPreadditiveAb …
    hxx : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    ⊢ Eq x 0
  -/
  obtain ⟨y, hy⟩ := KernelFork.IsLimit.lift' hl _ hxx
  /-
    case hf.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
    Z : C
    x : Quiver.Hom Z A
    hx : Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.NonPreadditiveAb …
    hxx : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    y : Quiver.Hom Z (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Limits. …
    hy : Eq (CategoryTheory.CategoryStruct.comp y (CategoryTheory.Limits.Fork.ι (C …
    ⊢ Eq x 0
  -/
  rw [KernelFork.ι_ofι] at hy
  have hyy : y = 0 := by
    erw [← Category.comp_id y, ← Limits.prod.lift_snd (𝟙 A) (𝟙 A), ← Category.assoc, hy,
      Category.assoc, prod.lift_snd, HasZeroMorphisms.comp_zero]
  /-
    case hf.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
    Z : C
    x : Quiver.Hom Z A
    hx : Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.NonPreadditiveAb …
    hxx : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    y : Quiver.Hom Z (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Limits. …
    hy : Eq (CategoryTheory.CategoryStruct.comp y (CategoryTheory.Limits.diag A))  …
    hyy : Eq y 0
    ⊢ Eq x 0
  -/
  haveI : Mono (prod.lift (𝟙 A) (0 : A ⟶ A)) := mono_of_mono_fac (prod.lift_fst _ _)
  /-
    case hf.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
    Z : C
    x : Quiver.Hom Z A
    hx : Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.NonPreadditiveAb …
    hxx : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    y : Quiver.Hom Z (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Limits. …
    hy : Eq (CategoryTheory.CategoryStruct.comp y (CategoryTheory.Limits.diag A))  …
    hyy : Eq y 0
    this : CategoryTheory.Mono (CategoryTheory.Limits.prod.lift (CategoryTheory.Ca …
    ⊢ Eq x 0
  -/
  apply (cancel_mono (prod.lift (𝟙 A) (0 : A ⟶ A))).1
  /-
    case hf.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
    Z : C
    x : Quiver.Hom Z A
    hx : Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.NonPreadditiveAb …
    hxx : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    y : Quiver.Hom Z (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Limits. …
    hy : Eq (CategoryTheory.CategoryStruct.comp y (CategoryTheory.Limits.diag A))  …
    hyy : Eq y 0
    this : CategoryTheory.Mono (CategoryTheory.Limits.prod.lift (CategoryTheory.Ca …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.Limits.prod.lift (C …
  -/
  rw [← hy, hyy, zero_comp, zero_comp]
  /-
    🎉 no goals
  -/


instance epi_r {A : C} : Epi (r A) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    ⊢ CategoryTheory.Epi (CategoryTheory.NonPreadditiveAbelian.r A)
  -/
  have hlp : prod.lift (𝟙 A) (0 : A ⟶ A) ≫ Limits.prod.snd = 0 := prod.lift_snd _ _
  let hp1 : IsLimit (KernelFork.ofι (prod.lift (𝟙 A) (0 : A ⟶ A)) hlp) := by
    refine Fork.IsLimit.mk _ (fun s => Fork.ι s ≫ Limits.prod.fst) ?_ ?_
    · intro s
      apply Limits.prod.hom_ext <;> simp
    · intro s m h
      haveI : Mono (prod.lift (𝟙 A) (0 : A ⟶ A)) := mono_of_mono_fac (prod.lift_fst _ _)
      apply (cancel_mono (prod.lift (𝟙 A) (0 : A ⟶ A))).1
      convert h
      apply Limits.prod.hom_ext <;> simp
  let hp2 : IsColimit (CokernelCofork.ofπ (Limits.prod.snd : A ⨯ A ⟶ A) hlp) :=
    epiIsCokernelOfKernel _ hp1
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hlp : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift  …
    hp1 : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cat …
    hp2 : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.of …
    ⊢ CategoryTheory.Epi (CategoryTheory.NonPreadditiveAbelian.r A)
  -/
  apply NormalMonoCategory.epi_of_zero_cancel
  /-
    case hf
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hlp : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift  …
    hp1 : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cat …
    hp2 : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.of …
    ⊢ ∀ (Z : C) (g : Quiver.Hom (CategoryTheory.Limits.cokernel (CategoryTheory.Li …
  -/
  intro Z z hz
  /-
    case hf
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hlp : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift  …
    hp1 : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cat …
    hp2 : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.of …
    Z : C
    z : Quiver.Hom (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.diag A)) Z
    hz : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NonPreadditiveAbel …
    ⊢ Eq z 0
  -/
  have h : prod.lift (𝟙 A) (0 : A ⟶ A) ≫ cokernel.π (diag A) ≫ z = 0 := by rw [← Category.assoc, hz]
  /-
    case hf
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hlp : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift  …
    hp1 : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cat …
    hp2 : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.of …
    Z : C
    z : Quiver.Hom (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.diag A)) Z
    hz : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NonPreadditiveAbel …
    h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (C …
    ⊢ Eq z 0
  -/
  obtain ⟨t, ht⟩ := CokernelCofork.IsColimit.desc' hp2 _ h
  /-
    case hf.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hlp : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift  …
    hp1 : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cat …
    hp2 : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.of …
    Z : C
    z : Quiver.Hom (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.diag A)) Z
    hz : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NonPreadditiveAbel …
    h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (C …
    t : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ CategoryTheory.Limits …
    ht : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (C …
    ⊢ Eq z 0
  -/
  rw [CokernelCofork.π_ofπ] at ht
  have htt : t = 0 := by
    rw [← Category.id_comp t]
    change 𝟙 A ≫ t = 0
    rw [← Limits.prod.lift_snd (𝟙 A) (𝟙 A), Category.assoc, ht, ← Category.assoc,
      cokernel.condition, zero_comp]
  /-
    case hf.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hlp : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift  …
    hp1 : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cat …
    hp2 : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.of …
    Z : C
    z : Quiver.Hom (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.diag A)) Z
    hz : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NonPreadditiveAbel …
    h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (C …
    t : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ CategoryTheory.Limits …
    ht : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.prod.snd t)  …
    htt : Eq t 0
    ⊢ Eq z 0
  -/
  apply (cancel_epi (cokernel.π (diag A))).1
  /-
    case hf.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    A : C
    hlp : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift  …
    hp1 : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cat …
    hp2 : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.of …
    Z : C
    z : Quiver.Hom (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.diag A)) Z
    hz : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NonPreadditiveAbel …
    h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (C …
    t : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ CategoryTheory.Limits …
    ht : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.prod.snd t)  …
    htt : Eq t 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π (Ca …
  -/
  rw [← ht, htt, comp_zero, comp_zero]
  /-
    🎉 no goals
  -/


instance isIso_r {A : C} : IsIso (r A) :=
  isIso_of_mono_of_epi _


/-- The composite `A ⨯ A ⟶ cokernel (diag A) ⟶ A` given by the natural projection into the cokernel
    followed by the inverse of `r`. In the category of modules, using the normal kernels and
    cokernels, this map is equal to the map `(a, b) ↦ a - b`, hence the name `σ` for
    "subtraction". -/
abbrev σ {A : C} : A ⨯ A ⟶ A :=
  cokernel.π (diag A) ≫ inv (r A)


@[reassoc]
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                inst✝ : CategoryTheory.NonPreadditiveAbelian C
                                                X : C
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diag X) Catego …
                                              -/
theorem diag_σ {X : C} : diag X ≫ σ = 0 := by rw [cokernel.condition_assoc, zero_comp]
                                              /-
                                                🎉 no goals
                                              -/


@[reassoc (attr := simp)]
                                                           /-
                                                             C : Type u
                                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                                             inst✝ : CategoryTheory.NonPreadditiveAbelian C
                                                             X : C
                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (Cat …
                                                           -/
theorem lift_σ {X : C} : prod.lift (𝟙 X) 0 ≫ σ = 𝟙 X := by rw [← Category.assoc, IsIso.hom_inv_id]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[reassoc]
theorem lift_map {X Y : C} (f : X ⟶ Y) :
                                                                          /-
                                                                            C : Type u
                                                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                            inst✝ : CategoryTheory.NonPreadditiveAbelian C
                                                                            X Y : C
                                                                            f : Quiver.Hom X Y
                                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (Cat …
                                                                          -/
    prod.lift (𝟙 X) 0 ≫ Limits.prod.map f f = f ≫ prod.lift (𝟙 Y) 0 := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- σ is a cokernel of Δ X. -/
def isColimitσ {X : C} : IsColimit (CokernelCofork.ofπ (σ : X ⨯ X ⟶ X) diag_σ) :=
                                                  /-
                                                    C : Type u
                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                    inst✝ : CategoryTheory.NonPreadditiveAbelian C
                                                    X : C
                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π (Ca …
                                                  -/
  cokernel.cokernelIso _ σ (asIso (r X)).symm (by rw [Iso.symm_hom, asIso_inv])
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- This is the key identity satisfied by `σ`. -/
theorem σ_comp {X Y : C} (f : X ⟶ Y) : σ ≫ f = Limits.prod.map f f ≫ σ := by
  obtain ⟨g, hg⟩ :=
    CokernelCofork.IsColimit.desc' isColimitσ (Limits.prod.map f f ≫ σ) (by
      rw [prod.diag_map_assoc, diag_σ, comp_zero])
  /-
    case mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    f : Quiver.Hom X Y
    g : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ CategoryTheory.NonPre …
    hg : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (C …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.NonPreadditiveAbelian. …
  -/
  suffices hfg : f = g by rw [← hg, Cofork.π_ofπ, hfg]
  calc
    f = f ≫ prod.lift (𝟙 Y) 0 ≫ σ := by rw [lift_σ, Category.comp_id]
    _ = prod.lift (𝟙 X) 0 ≫ Limits.prod.map f f ≫ σ := by rw [lift_map_assoc]
    _ = prod.lift (𝟙 X) 0 ≫ σ ≫ g := by rw [← hg, CokernelCofork.π_ofπ]
    _ = g := by rw [← Category.assoc, lift_σ, Category.id_comp]


/-- Subtraction of morphisms in a `NonPreadditiveAbelian` category. -/
def hasSub {X Y : C} : Sub (X ⟶ Y) :=
  ⟨fun f g => prod.lift f g ≫ σ⟩


/-- Negation of morphisms in a `NonPreadditiveAbelian` category. -/
def hasNeg {X Y : C} : Neg (X ⟶ Y) where
  neg := fun f => 0 - f


/-- Addition of morphisms in a `NonPreadditiveAbelian` category. -/
def hasAdd {X Y : C} : Add (X ⟶ Y) :=
  ⟨fun f g => f - -g⟩


theorem sub_def {X Y : C} (a b : X ⟶ Y) : a - b = prod.lift a b ≫ σ := rfl


theorem add_def {X Y : C} (a b : X ⟶ Y) : a + b = a - -b := rfl


theorem neg_def {X Y : C} (a : X ⟶ Y) : -a = 0 - a := rfl


theorem sub_zero {X Y : C} (a : X ⟶ Y) : a - 0 = a := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a : Quiver.Hom X Y
    ⊢ Eq (HSub.hSub a 0) a
  -/
  rw [sub_def]
  conv_lhs =>
    congr; congr; rw [← Category.comp_id a]
    case a.g => rw [show 0 = a ≫ (0 : Y ⟶ Y) by simp]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (Cat …
  -/
  rw [← prod.comp_lift, Category.assoc, lift_σ, Category.comp_id]
  /-
    🎉 no goals
  -/


theorem sub_self {X Y : C} (a : X ⟶ Y) : a - a = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a : Quiver.Hom X Y
    ⊢ Eq (HSub.hSub a a) 0
  -/
  rw [sub_def, ← Category.comp_id a, ← prod.comp_lift, Category.assoc, diag_σ, comp_zero]
  /-
    🎉 no goals
  -/


theorem lift_sub_lift {X Y : C} (a b c d : X ⟶ Y) :
    prod.lift a b - prod.lift c d = prod.lift (a - c) (b - d) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b c d : Quiver.Hom X Y
    ⊢ Eq (HSub.hSub (CategoryTheory.Limits.prod.lift a b) (CategoryTheory.Limits.p …
  -/
  simp only [sub_def]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b c d : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (Cat …
  -/
  ext
    /-
      case h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.NonPreadditiveAbelian C
      X Y : C
      a b c d : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · rw [Category.assoc, σ_comp, prod.lift_map_assoc, prod.lift_fst, prod.lift_fst, prod.lift_fst]
    /-
      🎉 no goals
    -/
    /-
      case h₂
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.NonPreadditiveAbelian C
      X Y : C
      a b c d : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · rw [Category.assoc, σ_comp, prod.lift_map_assoc, prod.lift_snd, prod.lift_snd, prod.lift_snd]
    /-
      🎉 no goals
    -/


theorem sub_sub_sub {X Y : C} (a b c d : X ⟶ Y) : a - c - (b - d) = a - b - (c - d) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b c d : Quiver.Hom X Y
    ⊢ Eq (HSub.hSub (HSub.hSub a c) (HSub.hSub b d)) (HSub.hSub (HSub.hSub a b) (H …
  -/
  rw [sub_def, ← lift_sub_lift, sub_def, Category.assoc, σ_comp, prod.lift_map_assoc]; rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


theorem neg_sub {X Y : C} (a b : X ⟶ Y) : -a - b = -b - a := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b : Quiver.Hom X Y
    ⊢ Eq (HSub.hSub (Neg.neg a) b) (HSub.hSub (Neg.neg b) a)
  -/
  conv_lhs => rw [neg_def, ← sub_zero b, sub_sub_sub, sub_zero, ← neg_def]
  /-
    🎉 no goals
  -/


theorem neg_neg {X Y : C} (a : X ⟶ Y) : - -a = a := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a : Quiver.Hom X Y
    ⊢ Eq (Neg.neg (Neg.neg a)) a
  -/
  rw [neg_def, neg_def]
  conv_lhs =>
    congr; rw [← sub_self a]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a : Quiver.Hom X Y
    ⊢ Eq (HSub.hSub (HSub.hSub a a) (HSub.hSub 0 a)) a
  -/
  rw [sub_sub_sub, sub_zero, sub_self, sub_zero]
  /-
    🎉 no goals
  -/


theorem add_comm {X Y : C} (a b : X ⟶ Y) : a + b = b + a := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b : Quiver.Hom X Y
    ⊢ Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
  -/
  rw [add_def]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b : Quiver.Hom X Y
    ⊢ Eq (HSub.hSub a (Neg.neg b)) (HAdd.hAdd b a)
  -/
  conv_lhs => rw [← neg_neg a]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b : Quiver.Hom X Y
    ⊢ Eq (HSub.hSub (Neg.neg (Neg.neg a)) (Neg.neg b)) (HAdd.hAdd b a)
  -/
  rw [neg_def, neg_def, neg_def, sub_sub_sub]
  conv_lhs =>
    congr
    next => skip
    rw [← neg_def, neg_sub]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b : Quiver.Hom X Y
    ⊢ Eq (HSub.hSub (HSub.hSub 0 0) (HSub.hSub (Neg.neg b) a)) (HAdd.hAdd b a)
  -/
  rw [sub_sub_sub, add_def, ← neg_def, neg_neg b, neg_def]
  /-
    🎉 no goals
  -/


                                                               /-
                                                                 C : Type u
                                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                 inst✝ : CategoryTheory.NonPreadditiveAbelian C
                                                                 X Y : C
                                                                 a b : Quiver.Hom X Y
                                                                 ⊢ Eq (HAdd.hAdd a (Neg.neg b)) (HSub.hSub a b)
                                                               -/
theorem add_neg {X Y : C} (a b : X ⟶ Y) : a + -b = a - b := by rw [add_def, neg_neg]
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                                /-
                                                                  C : Type u
                                                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                  inst✝ : CategoryTheory.NonPreadditiveAbelian C
                                                                  X Y : C
                                                                  a : Quiver.Hom X Y
                                                                  ⊢ Eq (HAdd.hAdd a (Neg.neg a)) 0
                                                                -/
theorem add_neg_cancel {X Y : C} (a : X ⟶ Y) : a + -a = 0 := by rw [add_neg, sub_self]
                                                                /-
                                                                  🎉 no goals
                                                                -/


                                                                /-
                                                                  C : Type u
                                                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                  inst✝ : CategoryTheory.NonPreadditiveAbelian C
                                                                  X Y : C
                                                                  a : Quiver.Hom X Y
                                                                  ⊢ Eq (HAdd.hAdd (Neg.neg a) a) 0
                                                                -/
theorem neg_add_cancel {X Y : C} (a : X ⟶ Y) : -a + a = 0 := by rw [add_comm, add_neg_cancel]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem neg_sub' {X Y : C} (a b : X ⟶ Y) : -(a - b) = -a + b := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b : Quiver.Hom X Y
    ⊢ Eq (Neg.neg (HSub.hSub a b)) (HAdd.hAdd (Neg.neg a) b)
  -/
  rw [neg_def, neg_def]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b : Quiver.Hom X Y
    ⊢ Eq (HSub.hSub 0 (HSub.hSub a b)) (HAdd.hAdd (HSub.hSub 0 a) b)
  -/
  conv_lhs => rw [← sub_self (0 : X ⟶ Y)]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b : Quiver.Hom X Y
    ⊢ Eq (HSub.hSub (HSub.hSub 0 0) (HSub.hSub a b)) (HAdd.hAdd (HSub.hSub 0 a) b)
  -/
  rw [sub_sub_sub, add_def, neg_def]
  /-
    🎉 no goals
  -/


                                                                  /-
                                                                    C : Type u
                                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                    inst✝ : CategoryTheory.NonPreadditiveAbelian C
                                                                    X Y : C
                                                                    a b : Quiver.Hom X Y
                                                                    ⊢ Eq (Neg.neg (HAdd.hAdd a b)) (HSub.hSub (Neg.neg a) b)
                                                                  -/
theorem neg_add {X Y : C} (a b : X ⟶ Y) : -(a + b) = -a - b := by rw [add_def, neg_sub', add_neg]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem sub_add {X Y : C} (a b c : X ⟶ Y) : a - b + c = a - (b - c) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b c : Quiver.Hom X Y
    ⊢ Eq (HAdd.hAdd (HSub.hSub a b) c) (HSub.hSub a (HSub.hSub b c))
  -/
  rw [add_def, neg_def, sub_sub_sub, sub_zero]
  /-
    🎉 no goals
  -/


theorem add_assoc {X Y : C} (a b c : X ⟶ Y) : a + b + c = a + (b + c) := by
  conv_lhs =>
    congr; rw [add_def]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y : C
    a b c : Quiver.Hom X Y
    ⊢ Eq (HAdd.hAdd (HSub.hSub a (Neg.neg b)) c) (HAdd.hAdd a (HAdd.hAdd b c))
  -/
  rw [sub_add, ← add_neg, neg_sub', neg_neg]
  /-
    🎉 no goals
  -/


                                                         /-
                                                           C : Type u
                                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                                           inst✝ : CategoryTheory.NonPreadditiveAbelian C
                                                           X Y : C
                                                           a : Quiver.Hom X Y
                                                           ⊢ Eq (HAdd.hAdd a 0) a
                                                         -/
theorem add_zero {X Y : C} (a : X ⟶ Y) : a + 0 = a := by rw [add_def, neg_def, sub_self, sub_zero]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem comp_sub {X Y Z : C} (f : X ⟶ Y) (g h : Y ⟶ Z) : f ≫ (g - h) = f ≫ g - f ≫ h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y Z : C
    f : Quiver.Hom X Y
    g h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HSub.hSub g h)) (HSub.hSub (Catego …
  -/
  rw [sub_def, ← Category.assoc, prod.comp_lift, sub_def]
  /-
    🎉 no goals
  -/


theorem sub_comp {X Y Z : C} (f g : X ⟶ Y) (h : Y ⟶ Z) : (f - g) ≫ h = f ≫ h - g ≫ h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y Z : C
    f g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub f g) h) (HSub.hSub (Catego …
  -/
  rw [sub_def, Category.assoc, σ_comp, ← Category.assoc, prod.lift_map, sub_def]
  /-
    🎉 no goals
  -/


theorem comp_add (X Y Z : C) (f : X ⟶ Y) (g h : Y ⟶ Z) : f ≫ (g + h) = f ≫ g + f ≫ h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y Z : C
    f : Quiver.Hom X Y
    g h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HAdd.hAdd g h)) (HAdd.hAdd (Catego …
  -/
  rw [add_def, comp_sub, neg_def, comp_sub, comp_zero, add_def, neg_def]
  /-
    🎉 no goals
  -/


theorem add_comp (X Y Z : C) (f g : X ⟶ Y) (h : Y ⟶ Z) : (f + g) ≫ h = f ≫ h + g ≫ h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.NonPreadditiveAbelian C
    X Y Z : C
    f g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f g) h) (HAdd.hAdd (Catego …
  -/
  rw [add_def, sub_comp, neg_def, sub_comp, zero_comp, add_def, neg_def]
  /-
    🎉 no goals
  -/


/-- Every `NonPreadditiveAbelian` category is preadditive. -/
def preadditive : Preadditive C where
  homGroup X Y :=
    { add := (· + ·)
      add_assoc := add_assoc
      zero := 0
      zero_add := neg_neg
      add_zero := add_zero
      neg := fun f => -f
      neg_add_cancel := neg_add_cancel
      sub_eq_add_neg := fun f g => (add_neg f g).symm -- Porting note: autoParam failed
      add_comm := add_comm
      nsmul := nsmulRec
      zsmul := zsmulRec }
  add_comp := add_comp
  comp_add := comp_add


