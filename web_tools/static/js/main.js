/**
 * Crisis MAS Web Tools - Client-Side JavaScript
 */

// Auto-hide flash messages after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    const alerts = document.querySelectorAll('.alert:not(.alert-info)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});

// Form validation enhancement
document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
});

// Confirm delete actions
function confirmDelete(itemType, itemName) {
    return confirm('Are you sure you want to delete this ' + itemType + ': "' + itemName + '"?\n\nThis action cannot be undone.');
}

// Smooth scroll to top
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Add scroll-to-top button if page is long
document.addEventListener('DOMContentLoaded', function() {
    if (document.body.scrollHeight > 2000) {
        const scrollBtn = document.createElement('button');
        scrollBtn.innerHTML = '<i class="bi bi-arrow-up"></i>';
        scrollBtn.className = 'btn btn-primary position-fixed';
        scrollBtn.style.cssText = 'bottom: 20px; right: 20px; z-index: 1000; display: none; border-radius: 50%; width: 50px; height: 50px;';
        scrollBtn.onclick = scrollToTop;
        document.body.appendChild(scrollBtn);
        
        window.addEventListener('scroll', function() {
            scrollBtn.style.display = window.pageYOffset > 300 ? 'block' : 'none';
        });
    }
});

// Tooltip initialization
document.addEventListener('DOMContentLoaded', function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Loading state for forms
document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn && form.checkValidity()) {
                submitBtn.disabled = true;
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Saving...';
                
                // Re-enable after 10 seconds as fallback
                setTimeout(() => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalText;
                }, 10000);
            }
        });
    });
});

console.log('Crisis MAS Web Tools - JavaScript loaded successfully');
